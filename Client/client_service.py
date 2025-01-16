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
    # log = logging.getLogger('producer')
    # log.info('Received processed message')
    serialized_message = json.dumps(message)
    logging.debug('serial ' + str(serialized_message))
    try:
        await websocket.send(serialized_message)
    except Exception as e:
        logging.debug('producer exception catch ' + str(e))


async def listener(websocket, path):
    if path == '/process':

        async for message in websocket:
            print('Starting local model updating')
            job_data = pickle.loads(message)
            await process(job_data, websocket)
            print('Model update completed. Sending results back to PS')
            await websocket.close()



if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser("client")
        parser.add_argument("port", help="Define a valid port for the  client to run on", type=int)
        args = parser.parse_args()
        print('client running on ' + str(args.port))
        start_server = websockets.serve(listener, "0.0.0.0", args.port, ping_timeout=None, max_size=None)
        loop = asyncio.get_event_loop()

        loop.run_until_complete(start_server)
        loop.run_forever()
    except Exception as e:
        print(f'Caught exception {e}')
        pass
    finally:
        loop.close()
