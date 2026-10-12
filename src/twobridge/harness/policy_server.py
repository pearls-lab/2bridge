"""Serve a release checkpoint over the OpenPI websocket wire protocol."""

import argparse
import asyncio
import logging

logger = logging.getLogger(__name__)


def make_handler(policy):
    from websockets.exceptions import ConnectionClosed

    from .wire import pack, unpack

    async def handler(socket):
        await socket.send(pack({"implementation": "twobridge", "action_dim": policy.action_dim}))
        try:
            async for message in socket:
                observation = unpack(message)
                prompt = observation.pop("prompt")
                # Synchronous inference serializes access to the GPU across connections.
                actions = policy.predict(observation, prompt)
                await socket.send(pack({"actions": actions}))
        except ConnectionClosed:
            pass
        except Exception as exc:
            logger.exception("Policy inference failed")
            await socket.send(f"Policy inference failed: {type(exc).__name__}: {exc}")
            await socket.close(code=1011)

    return handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    from websockets.asyncio.server import serve

    from twobridge.policy import Policy

    policy = Policy(args.checkpoint, args.device)

    async def run():
        async with serve(make_handler(policy), args.host, args.port, compression=None,
                         max_size=32 * 1024 * 1024, ping_interval=None) as server:
            await server.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
