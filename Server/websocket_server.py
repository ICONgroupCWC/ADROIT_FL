import asyncio
import websockets
import json
import argparse
from processors.server_start_process import JobServer
from utils.wireless_utils import calculate_avg_bit_rate


class WebSocketServer:
    def __init__(self, num_ues):
        self.avg_bitrate = calculate_avg_bit_rate(num_ues)


    async def listener(self, websocket):
        print(f"New connection on path: {websocket}")  # Debug print
        if websocket.request.path == '/job_receive':
            async for message in websocket:
                print('received a request for new FL task')

                job_data = json.loads(message)
                # Add the bitrate parameter to job_data
                job_data['avg_bitrate'] = self.avg_bitrate

                local_loop = asyncio.get_running_loop()
                job_server = JobServer()
                local_loop.create_task(job_server.start_job(job_data, websocket))

    async def start(self):
        try:

            server = await websockets.serve(self.listener, "0.0.0.0", 8200, ping_interval=None)

            await server.wait_closed()
            print("Server closed")  # Debug print
        except Exception as e:
            print(f'Caught exception {e}')
            raise  # Re-raise the exception to see the full traceback


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start WebSocket server with specified number of ues')
    parser.add_argument('--num_ues', type=int, required=True, help='Number of ues connected to the central server')

    args = parser.parse_args()
    server = WebSocketServer(args.num_ues)
    asyncio.run(server.start())