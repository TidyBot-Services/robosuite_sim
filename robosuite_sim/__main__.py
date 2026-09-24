"""Start with: python -m robosuite_sim --port 8082."""

from __future__ import annotations

import argparse

from .server import SimulatorState, create_server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--enable-sim-gt", action="store_true",
        help="enable the separate v2 GT-perception track; never use for v1 non-oracle scores",
    )
    args = parser.parse_args()
    server = create_server(args.host, args.port, SimulatorState(enable_sim_gt=args.enable_sim_gt))
    host, port = server.server_address
    print(f"robosuite_sim listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
