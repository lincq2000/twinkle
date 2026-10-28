# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression tests for ``twinkle.utils.torch_utils.stateless_init_process_group``.

The rank-0 TCPStore serves on a listening socket created in Python. Older vLLM keeps that socket
alive through a ``StatelessProcessGroup.socket`` field; newer vLLM removed the field. If the socket
object is garbage collected while the store still uses its fd, the store's server thread aborts
the whole process (SIGABRT), so every case runs in a subprocess.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent('''
    import dataclasses
    import gc
    import socket
    import sys
    import threading
    import types
    from datetime import timedelta

    from torch.distributed import TCPStore

    with_socket_field = sys.argv[1] == 'with_socket_field'

    @dataclasses.dataclass
    class StatelessProcessGroup:
        rank: int
        world_size: int
        store: object
        data_expiration_seconds: int = 3600

    if with_socket_field:

        @dataclasses.dataclass
        class StatelessProcessGroup(StatelessProcessGroup):  # noqa: F811
            socket: object = None

    class PyNcclCommunicator:

        def __init__(self, group, device):
            self.group = group

    names = ('vllm', 'vllm.distributed', 'vllm.distributed.utils', 'vllm.distributed.device_communicators',
             'vllm.distributed.device_communicators.pynccl')
    modules = {name: types.ModuleType(name) for name in names}
    modules['vllm.distributed.utils'].StatelessProcessGroup = StatelessProcessGroup
    modules['vllm.distributed.device_communicators.pynccl'].PyNcclCommunicator = PyNcclCommunicator
    sys.modules.update(modules)

    from twinkle.utils.torch_utils import stateless_init_process_group

    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]

    def init(rank):
        return stateless_init_process_group('127.0.0.1', port, rank=rank, world_size=2, device=0, backend='nccl')

    results = {}
    worker = threading.Thread(target=lambda: results.update(worker=init(1)))
    worker.start()
    master = init(0)
    worker.join()
    gc.collect()

    master.group.store.set('key', b'value')
    assert results['worker'].group.store.get('key') == b'value'
    # A fresh connection proves the listening socket is still open.
    late = TCPStore('127.0.0.1', port, is_master=False, timeout=timedelta(seconds=10))
    assert late.get('key') == b'value'
    print('ok')
''')


@pytest.mark.parametrize('variant', ['with_socket_field', 'without_socket_field'])
def test_stateless_init_process_group_keeps_store_alive(variant):
    result = subprocess.run([sys.executable, '-c', _SCRIPT, variant],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            env=os.environ.copy())
    assert result.returncode == 0 and result.stdout.strip().endswith('ok'), (
        f'exit {result.returncode}\nstdout:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-4000:]}')
