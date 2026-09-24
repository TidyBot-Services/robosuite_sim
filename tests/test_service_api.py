from __future__ import annotations

import threading

import numpy as np

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
