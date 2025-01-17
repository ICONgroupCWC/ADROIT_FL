import asyncio
import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor
import websockets
from processors.client_process import process
import json
import argparse

task_executor = ProcessPoolExecutor(max_workers=3)


async def producer(websocket, message):
    serialized_message = json.dumps(message)
    logging.debug('serial ' + str(serialized_message))
    try:
        await websocket.send(serialized_message)
    except Exception as e:
        logging.debug('producer exception catch ' + str(e))


async def listener(websocket):
    if websocket.request.path == '/process':
        async for message in websocket:
            print('Starting local model updating')
            job_data = pickle.loads(message)
            await process(job_data, websocket)
            print('Model update completed. Sending results back to PS')
            await websocket.close()


async def start_client(port):
    try:
        print('client running on ' + str(port))
        server = await websockets.serve(listener, "0.0.0.0", port, ping_timeout=None, max_size=None)
        await server.wait_closed()
    except Exception as e:
        print(f'Caught exception {e}')
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="client")
        parser.add_argument("port", help="Define a valid port for the client to run on", type=int)
        args = parser.parse_args()

        asyncio.run(start_client(args.port))
    except Exception as e:
        print(f'Caught exception {e}')