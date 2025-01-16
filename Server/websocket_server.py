import asyncio
import websockets
from Server.processors.server_start_process import JobServer

import json



async def listener(websocket, path):

    if path == '/job_receive':

        async for message in websocket:
            print('received a request for new FL task')

            job_data = json.loads(message)

            # job_data = json.loads(ms['jobData'])
            local_loop = asyncio.get_running_loop()
            # await start_job(job_data, websocket)
            job_server = JobServer()
            local_loop.create_task(job_server.start_job(job_data, websocket))

            # job_server.start_job(job_data)


try:
    print('starting the PS server...')
    start_server = websockets.serve(listener, "0.0.0.0", 8200, ping_interval=None)
    loop = asyncio.get_event_loop()

    loop.run_until_complete(start_server)
    print('PS server started and running...')
    loop.run_forever()
except Exception as e:
    print(f'Caught exception {e}')
    pass
finally:
    loop.close()
