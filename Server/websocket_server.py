import asyncio
import websockets
import json
import argparse
from Server.processors.server_start_process import JobServer
from Server.utils.wireless_utils import calculate_avg_bit_rate


class WebSocketServer:
    def __init__(self, num_ues):
        self.avg_bitrate = calculate_avg_bit_rate(num_ues)
        print(f"Initialized server with {num_ues} users. Average bitrate: {self.avg_bitrate}")

    async def listener(self, websocket, path):
        if path == '/job_receive':
            async for message in websocket:
                print('received a request for new FL task')

                job_data = json.loads(message)
                # Add the bitrate parameter to job_data
                job_data['avg_bitrate'] = self.avg_bitrate

                local_loop = asyncio.get_running_loop()
                job_server = JobServer()
                local_loop.create_task(job_server.start_job(job_data, websocket))

    def start(self):
        try:
            print('starting the PS server...')
            start_server = websockets.serve(self.listener, "0.0.0.0", 8200, ping_interval=None)
            loop = asyncio.get_event_loop()

            loop.run_until_complete(start_server)
            print('PS server started and running...')
            loop.run_forever()
        except Exception as e:
            print(f'Caught exception {e}')
        finally:
            loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start WebSocket server with specified number of ues')
    parser.add_argument('--num_ues', type=int, required=True, help='Number of ues connected to the central server')

    args = parser.parse_args()

    server = WebSocketServer(args.num_ues)
    server.start()