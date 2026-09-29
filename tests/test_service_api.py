from __future__ import annotations

import threading
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robosuite_sim.backend import BackendConfig, RobosuiteBackend
from robosuite_sim.client import RobosuiteSimClient
from robosuite_sim.server import SimulatorState, create_server
from robosuite_sim.tasks import get_task


class FakeBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.spec = get_task(config.task_id)
        self.action_spec = (np.full(7, -1.0), np.full(7, 1.0))
        self.position = np.zeros(3)
        self.trace = []

    @property
    def metadata(self):
        return {"service": "robosuite_sim", "task_id": self.spec.task_id}

    def _observation(self):
        return {"robot0_eef_pos": self.position.copy()}

    def reset(self, _seed):
        self.position[:] = 0
        self.trace.clear()
        return self._observation()

    def step(self, action):
        self.position += np.asarray(action[:3]) * 0.05
        self.trace.append({"step": len(self.trace), "action": np.asarray(action).tolist()})
        return self._observation(), 0.0, False, {}

    def observe(self):
        return self._observation()

    def native_success(self):
        return bool(self.position[2] > 0)

    def variation_identity(self):
        return {"scene_id": "scene:fixed", "object_set_id": "objects:fixed"}

    def gt_objects(self):
        return [{"name": "cube", "position": [0.1, 0.2, 0.3]}]

    def run_reference(self, _timeout):
        self.step(np.array([0, 0, 1, 0, 0, 0, -1], dtype=float))
        return self._observation()

    def close(self):
        pass


def test_public_filter_removes_object_oracle() -> None:
    public = RobosuiteBackend._filter_observation(
        {
            "agentview_image": np.zeros((2, 2, 3)),
            "robot0_joint_pos": np.zeros(7),
            "cube_pos": np.ones(3),
            "object-state": np.ones(4),
        }
    )
    assert set(public) == {"agentview_image", "robot0_joint_pos"}


def test_metric_depth_rejects_and_describes_invalid_frames(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TIDYBOT_INVALID_DEPTH_DIR", str(tmp_path))
    sim = SimpleNamespace(model=SimpleNamespace(
        stat=SimpleNamespace(extent=1.0),
        vis=SimpleNamespace(map=SimpleNamespace(znear=0.1, zfar=10.0)),
    ))
    for frame, counts in (
        (np.array([[0.0, 1.0 + 1e-7]], dtype=np.float32), (0, 1, 0)),
        (np.array([[-0.01, np.nan]], dtype=np.float32), (1, 0, 1)),
    ):
        try:
            RobosuiteBackend._metric_depth(sim, frame)
        except ValueError as exc:
            details = json.loads(str(exc).split(": ", 1)[1])
            assert (details["below_zero_count"], details["above_one_count"],
                    details["nonfinite_count"]) == counts
            assert details["shape"] == [1, 2]
            assert len(details["frame_sha256"]) == 64
            captured = np.load(details["capture_path"], allow_pickle=False)
            np.testing.assert_equal(captured, frame)
            assert details["capture_sha256"] == hashlib.sha256(
                Path(details["capture_path"]).read_bytes()
            ).hexdigest()
            assert details["first_invalid_kind"] in {"above_one", "below_zero"}
        else:
            raise AssertionError("invalid depth frame was accepted")
    assert np.isfinite(RobosuiteBackend._metric_depth(
        sim, np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    )).all()


def test_repeated_invalid_frames_keep_both_captures_and_action_context(tmp_path, monkeypatch):
    monkeypatch.setenv("TIDYBOT_INVALID_DEPTH_DIR", str(tmp_path))
    backend = object.__new__(RobosuiteBackend)
    backend.config = BackendConfig(task_id="cube_lift")
    backend._env = SimpleNamespace(sim=object())
    backend.trace = []
    frame = np.array([[[1.0 + 1e-7]]], dtype=np.float32)
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    paths = []
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid normalized depth frame") as error:
            backend._public_observation({"agentview_depth": frame},
                                        phase="step", action=action)
        details = json.loads(str(error.value).split(": ", 1)[1])
        assert details["observation_phase"] == "step"
        assert details["backend_step_index"] == 0
        assert details["camera_name"] == "agentview"
        assert details["action_sha256"] == hashlib.sha256(action.tobytes()).hexdigest()
        paths.append(Path(details["capture_path"]))
    assert paths[0] != paths[1]
    assert all(path.exists() for path in paths)


@pytest.mark.parametrize("repeat_value,expected", [(0.5, "valid"), (1.1, "invalid")])
def test_depth_repeat_probe_preserves_failed_step_and_physics_state(
    tmp_path, monkeypatch, repeat_value, expected,
):
    monkeypatch.setenv("TIDYBOT_INVALID_DEPTH_DIR", str(tmp_path))
    sim = SimpleNamespace(
        model=SimpleNamespace(
            stat=SimpleNamespace(extent=1.0),
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.1, zfar=10.0)),
            camera_name2id=lambda _name: 0,
            cam_fovy=np.array([45.0]),
        ),
        data=SimpleNamespace(
            time=0.0, qpos=np.array([0.0]),
            cam_xmat=np.eye(3).reshape(1, 9), cam_xpos=np.zeros((1, 3)),
        ),
    )

    class StepOnceEnvironment:
        action_spec = (np.full(7, -1.0), np.full(7, 1.0))

        def __init__(self):
            self.sim = sim
            self.action_calls = 0
            self.observation_calls = 0

        def step(self, _action):
            self.action_calls += 1
            self.sim.data.time += 0.1
            self.sim.data.qpos[0] += 0.2
            return {"agentview_depth": np.array([[[1.1]]], dtype=np.float32)}, 0.0, False, {}

        def _get_observations(self, *, force_update):
            assert force_update is True
            self.observation_calls += 1
            return {"agentview_depth": np.array([[[repeat_value]]], dtype=np.float32)}

    backend = object.__new__(RobosuiteBackend)
    backend.config = BackendConfig(task_id="cube_lift")
    backend._env = StepOnceEnvironment()
    backend._last_raw = None
    backend.trace = []
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    if expected == "valid":
        public, _, _, info = backend.step(action)
        assert np.isfinite(public["agentview_depth"]).all()
        assert info["depth_recovered_without_physics_step"] is True
        details = backend.trace[0]["depth_recovery"]
        assert info["depth_recovery"]["first_frame_sha256"] == details["frame_sha256"]
        assert (info["depth_recovery"]["repeat_frame_sha256"]
                == details["repeat_observation"]["frame_sha256"])
    else:
        with pytest.raises(ValueError, match="invalid normalized depth frame") as error:
            backend.step(action)
        details = json.loads(str(error.value).split(": ", 1)[1])
    repeat = details["repeat_observation"]
    assert repeat["status"] == expected
    assert repeat["simulation_time_unchanged"] is True
    assert repeat["qpos_unchanged"] is True
    assert repeat["observation_only"] is True
    assert np.load(repeat["capture_path"], allow_pickle=False).item() == pytest.approx(repeat_value)
    assert backend._env.action_calls == 1
    assert backend._env.observation_calls == 1
    assert len(backend.trace) == 1
    assert backend.trace[0]["action_executed"] is True


def test_http_service_contract_round_trip() -> None:
    state = SimulatorState(backend_factory=FakeBackend)  # type: ignore[arg-type]
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = RobosuiteSimClient(f"http://{host}:{port}")
    try:
        assert client.health()["status"] == "ok"
        capabilities = client.capabilities()
        assert capabilities["api_version"] == "v1"
        assert capabilities["object_oracle_visible"] is False
        assert capabilities["policy_operations"] == ["attach", "observe", "step"]
        assert "success" not in capabilities["policy_operations"]
        assert "camera_intrinsics" in capabilities["sensors"]
        observation, low, high, metadata = client.reset(
            task_id="cube_lift", seed=101, camera=False
        )
        assert observation["robot0_eef_pos"].tolist() == [0.0, 0.0, 0.0]
        assert low.shape == high.shape == (7,)
        assert metadata["service"] == "robosuite_sim"
        attached, attached_low, attached_high, attached_metadata = client.attach()
        assert attached["robot0_eef_pos"].tolist() == [0.0, 0.0, 0.0]
        assert attached_low.shape == attached_high.shape == (7,)
        assert attached_metadata["task_id"] == "cube_lift"
        client.step(np.array([0, 0, 1, 0, 0, 0, -1], dtype=float))
        assert client.native_success() is True
        reference = client.run_reference(1.0)
        assert reference.trace
        client.close_environment()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_v2_gt_endpoint_is_explicit_and_reset_is_attested() -> None:
    state = SimulatorState(backend_factory=FakeBackend, enable_sim_gt=True)  # type: ignore[arg-type]
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = RobosuiteSimClient(f"http://{host}:{port}")
    try:
        _, _, _, _, applied = client.reset_attested(
            task_id="cube_lift", seed=101, camera_name="frontview",
            discover_variation=True,
        )
        assert applied == {"scene_id": "scene:fixed", "object_set_id": "objects:fixed"}
        assert client.perceive_gt(camera_names=["frontview"])["objects"] == [
            {"name": "cube", "position": [0.1, 0.2, 0.3]}
        ]
        assert client.perceive_gt(target_names=["missing"], camera_names=["frontview"])["objects"] == []
        assert state.capabilities()["object_oracle_visible"] is False
        try:
            client.perceive_gt(camera_names=["agentview"])
        except Exception:
            pass
        else:
            raise AssertionError("GT endpoint accepted the wrong active camera")
        try:
            client.reset_attested(
                task_id="cube_lift", seed=101,
                variation={"scene_id": "wrong", "object_set_id": "objects:fixed"},
            )
        except Exception:
            pass
        else:
            raise AssertionError("reset accepted a different realized variation")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_v1_default_does_not_expose_gt() -> None:
    state = SimulatorState(backend_factory=FakeBackend)  # type: ignore[arg-type]
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = RobosuiteSimClient(f"http://{host}:{port}")
    try:
        client.reset(task_id="cube_lift", seed=101)
        from robosuite_sim.client import ServiceError
        try:
            client.perceive_gt()
        except ServiceError:
            pass
        else:
            raise AssertionError("default v1 server exposed the GT endpoint")
        try:
            client.reset_attested(task_id="cube_lift", seed=101, discover_variation=True)
        except ServiceError:
            pass
        else:
            raise AssertionError("default v1 server exposed GT variation identity")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
