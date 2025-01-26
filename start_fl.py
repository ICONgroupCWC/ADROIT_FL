import asyncio
import os

import websockets
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import yaml


async def start_fl():
    with open('network_config.yml', 'r') as file:
        network_data = yaml.safe_load(file)

    host_ip = network_data['host']
    print(network_data['clients'])
    uri = "ws://" + str(host_ip) + "/job_receive"
    lr = 0.001
    task_name = 'mnist_fl'
    epochs = 1
    minibatch = 64
    client_fraction = 0.7
    minibatch_test = 4096
    comm_rounds = 20
    optimizer = 'Adam'
    loss = 'Huber'
    folder = 'satellite'
    async with websockets.connect(uri) as websocket:

        job_data = {"jobData": {
            "general": {"task": str(task_name), "algo": "regression",
                        "host": str(host_ip),
                        "clients": network_data['clients'],
                        },
            "scheme": {"minibatch": str(minibatch), "epoch": str(epochs),
                       "lr": str(lr), "scheduler": "random", "clientFraction": str(client_fraction),
                       "minibatchtest": str(minibatch_test),
                       "comRounds": str(comm_rounds)},
            "modelParam": {"optimizer": str(optimizer), "loss": str(loss), "compress": 'quantize', 'z_point': 0.0,
                           'scale': 0.1, 'num_bits': 16},
            "preprocessing": {"dtype": "regression", "folder": str(folder), "testfolder": str(folder),
                              "normalize": False}}}

        job_data = json.dumps(job_data)
        await websocket.send(job_data)

        test_accuracy = []
        train_loss = []
        test_loss = []
        round_time = []
        total_bytes = []
        final_round = False
        while not final_round:
            async for message in websocket:

                message = json.loads(message)

                if message['status'] == 'training':
                    print('Training epoch ' + str(message['epoch']) + ' completed at ' + str(message['client_id']))
                elif message['status'] == 'results':
                    if not message['final']:
                        print('Communication round ' + str(message['round']) + ' completed with accuracy ' + str(
                            message['accuracy']))
                    else:
                        print('Final communication round completed with accuracy ' + str(message['accuracy']))
                        print(f"Energy Required to Transmit the Model: {float(message['energy_per_round']):.2f} Joules per round")

                if message['status'] == 'results':
                    test_accuracy.append(float(message['accuracy']))
                    train_loss.append(float(message['train_loss']))
                    test_loss.append(float(message['test_loss']))
                    if len(round_time) == 0:
                        round_time.append(float(message['round_time']))
                        total_bytes.append(float(message['total_bytes']))
                    else:
                        r_time = round_time[-1] + float(message['round_time'])
                        t_bytes = total_bytes[-1] + float(message['total_bytes'])
                        round_time.append(r_time)
                        total_bytes.append(t_bytes)

                    final_round = message['final']
                    print(f'total elapsed time {float(round_time[-1]):.2f}')

                if final_round:
                    rounds = np.array([i for i in range(1, len(test_accuracy) + 1)])

                    font = {
                        'weight': 'bold',
                        'size': 20}

                    matplotlib.rc('font', **font)
                    # print('Current test accuracy ' + str(test_accuracy))
                    fig, axs = plt.subplots(2, figsize=(30, 20))
                    plt.subplots_adjust(hspace=0.3)
                    axs[0].plot(rounds, np.array(test_accuracy), color='r', linewidth=3.0, label='test accuracy')
                    axs[0].set_title('Number of rounds vs Test Accuracy')
                    axs[0].set(xlabel='Number of Rounds', ylabel='Test Accuracy')
                    axs[0].legend()

                    axs[1].plot(rounds, np.array(train_loss), color='b', linewidth=3.0, label='train loss')
                    axs[1].set_title('Number of rounds vs Train Loss')
                    axs[1].set(xlabel='Number of Rounds', ylabel='Train Loss')

                    axs[1].plot(rounds, np.array(test_loss), color='y', linewidth=3.0, label='test loss')
                    axs[1].set_title('Number of rounds vs Test Loss')
                    axs[1].set(xlabel='Number of Rounds', ylabel='Test Loss')

                    axs[1].legend()
                    plt.savefig('results.png')


if __name__ == "__main__":
    asyncio.run(start_fl())
