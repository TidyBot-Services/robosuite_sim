import json
import threading
from pathlib import Path

import numpy as np
import pytest

from robosuite_sim.codec import encode_array
from robosuite_sim.client import RobosuiteSimClient, ServiceError
from robosuite_sim.server import SimulatorState, create_server, StructuredServiceError
from test_service_api import FakeBackend


@pytest.mark.parametrize('failure', [False, True])
def test_http_replay_persists_executed_result_and_survives_restart(tmp_path, failure):
    class ReceiptBackend(FakeBackend):
        calls = 0
        def step(self, action):
            type(self).calls += 1
            observation, reward, done, info = super().step(action)
            entry = {'action_executed': True, 'action': action.tolist(), 'reward': 2.5, 'done': True}
            self._receipt_hook(entry)
            # The receipt must already be on disk before conversion can fail.
            saved = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
            assert saved[0]['receipt'] == entry
            if failure:
                raise ValueError('invalid normalized depth frame: ' + json.dumps({'repeat_observation': {'status': 'invalid'}}))
            return observation, 2.5, True, info
    state = SimulatorState(ReceiptBackend, receipt_dir=tmp_path)
    server = create_server('127.0.0.1', 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = RobosuiteSimClient('http://%s:%s' % server.server_address)
    action = np.zeros(7)
    try:
        client.reset(task_id='cube_lift', seed=9001)
        request = {'session_id': client.session_id, 'request_id': 'one', 'action': encode_array(action)}
        if failure:
            errors = []
            for _ in range(2):
                with pytest.raises(ServiceError) as err:
                    client.step(action, request_id='one')
                errors.append(err.value.payload)
            assert errors[0] == errors[1]
            assert errors[0]['action_executed'] is True
            assert errors[0]['receipt']['reward'] == 2.5
            assert errors[0]['receipt']['done'] is True
            assert errors[0]['stop']['environment_closed'] is True
            assert state.health()['active'] is False
        else:
            a = client.step(action, request_id='one')
            b = client.step(action, request_id='one')
            assert a.reward == b.reward == 2.5
        assert ReceiptBackend.calls == 1
        restarted = SimulatorState(ReceiptBackend, receipt_dir=tmp_path)
        if failure:
            with pytest.raises(StructuredServiceError) as err:
                restarted.step(request)
            assert err.value.payload == errors[0]
        else:
            assert restarted.step(request)['reward'] == 2.5
        request['action'] = encode_array(np.ones(7))
        with pytest.raises(StructuredServiceError) as err:
            restarted.step(request)
        assert err.value.status == 409
        assert ReceiptBackend.calls == 1
    finally:
        server.shutdown(); server.server_close(); thread.join(3)


def test_unresolved_intent_never_reexecutes(tmp_path):
    state = SimulatorState(FakeBackend, receipt_dir=tmp_path)
    state.reset({'task_id': 'cube_lift', 'seed': 9001})
    request = {'session_id': state._session_id, 'request_id': 'crash', 'action': encode_array(np.zeros(7))}
    import hashlib
    action_sha = hashlib.sha256(json.dumps(request['action'], sort_keys=True).encode()).hexdigest()
    state.journal.write({**request, 'phase': 'intent', 'action_sha256': action_sha, 'action_executed': None})
    with pytest.raises(StructuredServiceError) as err:
        state.step(request)
    assert err.value.status == 409
    assert state._backend.trace == []
