import os
import uuid
import websockets
import asyncio
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import copy
import numpy as np
import importlib
import inspect
from pathlib import Path
import sys
from utils.modelUtil import dequantize_tensor, decompress_tensor
import pickle
from DataLoaders.loaderUtil import getDataloader
from utils.message_utils import create_message, create_message_results
from utils.modelUtil import get_criterion
from utils.Scheduler import Scheduler
from utils.wireless_utils import calculate_energy
import time


class JobServer:

    def __init__(self):
        self.new_latencies = None
        self.num_clients = 0
        self.local_weights = []
        self.local_loss = []
        self.q_diff, self.info = [], []
        self.v, self.i, self.s = [], [], []
        self.bytes = []
        self.comp_len = 0
        self.cumulative_size_uploaded_Mbytes = 0
        self.cumulative_size_uploaded_Mbytes_array = []

    def load_dataset(self, folder):

        data_test = np.load('data/' + str(folder) + '/X.npy')
        labels = np.load('data/' + str(folder) + '/y.npy')
        return data_test, labels

    def testing(self, model, preprocessing, bs, criterion):

        dataset, labels = self.load_dataset(preprocessing['folder'])
        test_loss = 0
        correct = 0
        total_samples = 0
        test_loader = DataLoader(getDataloader(dataset, labels, preprocessing), batch_size=bs, shuffle=False)
        model.eval()
        with torch.no_grad():
            for data, label in test_loader:
                output = model(data)
                # label = np.squeeze(label)
                loss = criterion(output, label)
                test_loss += loss.item() * data.size(0)
                # if preprocessing['dtype'] != 'regression':
                _, pred = torch.max(output, 1)

                # correct += pred.eq(label.data.view_as(pred)).sum().item()
                correct += (pred == label).sum().item()
                total_samples += label.size(0)

            test_loss /= len(test_loader.dataset)
            if preprocessing['dtype'] != 'regression':
                test_accuracy = 100. * correct / total_samples
            else:

                predicted_values = output.detach().cpu().numpy()
                actual_values = label.detach().cpu().numpy()
                # Round the values
                predicted_int = np.rint(predicted_values)
                y_test_int = np.rint(actual_values)
                # Calculate matches
                matches = np.sum(predicted_int == y_test_int)
                total = len(y_test_int)

                print(f"Matches: {matches}, Total: {total}")
                test_accuracy = (matches / total) * 100

        return test_loss, test_accuracy

    async def connector(self, client_uri, data, client_index, server_socket):
        """connector function for connecting the server to the clients. This function is called asynchronously to
        1. send process requests to each client
        2. calculate local weights for each client separately"""

        async with websockets.connect(client_uri, ping_interval=None, max_size=3000000) as websocket:
            finished = False
            try:
                await websocket.send(data)
                start = time.time()
                while not finished:
                    async for message in websocket:
                        self.cumulative_size_uploaded_Mbytes += (sys.getsizeof(
                            message) / 1e6)  # sys.getsizeof() return the size of an object in bytes
                        try:
                            data = pickle.loads(message)
                            self.bytes.append(len(message))
                            if len(data) == 2:
                                self.local_weights.append(copy.deepcopy(data[0]))
                                self.local_loss.append(copy.deepcopy(data[1]))

                            elif len(data) == 3:
                                self.q_diff.append(copy.deepcopy(data[0]))
                                self.local_loss.append(copy.deepcopy(data[1]))
                                self.info.append(copy.deepcopy(data[2]))
                                self.comp_len = len(self.q_diff)

                            elif len(data) == 4:
                                self.v.append(copy.deepcopy(data[0]))
                                self.i.append(copy.deepcopy(data[1]))
                                self.s.append(copy.deepcopy(data[2]))
                                self.local_loss.append(copy.deepcopy(data[3]))
                                self.comp_len = len(self.v)

                            finished = True
                            self.new_latencies[0, client_index] = time.time() - start

                            break

                        except Exception as e:

                            await server_socket.send(message)

            except Exception as e:

                pass

    async def start_job(self, data, websocket):

        print('Initial model deployment at PS')
        global model

        job_id = uuid.uuid4().hex
        filename = './ModelData/Model.py'

        # os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'rb') as f:
            data['file'] = f.read()

        path_pyfile = Path(filename)
        sys.path.append(str(path_pyfile.parent))
        mod_path = str(path_pyfile).replace(os.path.sep, '.').strip('.py')
        imp_path = importlib.import_module(mod_path)

        for name_local in dir(imp_path):

            if inspect.isclass(getattr(imp_path, name_local)):
                modelClass = getattr(imp_path, name_local)
                model = modelClass()

        job_data = data['jobData']
        schemeData = job_data['scheme']
        client_list = job_data['general']['clients']
        T = int(schemeData['comRounds'])
        C = float(schemeData['clientFraction']) if 'clientFraction' in schemeData else 1
        schemeData['clientFraction'] = C
        K = int(len(client_list))
        E = int(schemeData['epoch'])
        eta = float(schemeData['lr'])
        B = int(schemeData['minibatch'])
        B_test = int(schemeData['minibatchtest'])
        preprocessing = job_data['preprocessing']
        compress = job_data['modelParam']['compress']
        scheduler_type = schemeData['scheduler']

        latency_avg = int(schemeData['latency_avg']) if scheduler_type == 'latency' else 1
        # db_service.save_job_data(job_data, job_id)

        criterion = get_criterion(job_data['modelParam']['loss'])

        global_weights = model.state_dict()
        train_loss = []
        test_loss = []
        test_accuracy = []
        round_times = []
        total_bytes = []
        # m = max(int(C * K), 1)
        print('Broadcasting the initial model and hyperparameters to clients')
        # run for number of communication rounds
        scheduler = Scheduler(scheduler_type, K, C, avg_rounds=latency_avg)
        for curr_round in range(1, T + 1):
            start_time = time.time()
            # TODO need to check
            print('---------------------')
            print('Current communication round: ' + str(curr_round))
            # S_t = np.random.choice(range(K), m, replace=False)
            S_t = scheduler.get_workers(self.new_latencies)
            self.new_latencies = np.ones((1, K), dtype='float')
            client_ports = [clt for clt in client_list]
            clients = [client_ports[i] for i in S_t]
            st_count = 0

            tasks = []
            if curr_round > 0:
                print('1- PS Broadcasting global model to the clients')
            print('2- Local model training at clients\' side started')
            for client in clients:
                client_uri = 'ws://' + str(client['client_ip']) + '/process'
                serialized_data = create_message(B, eta, E, data['file'], job_data['modelParam'],
                                                 preprocessing, global_weights)
                client_index = client_ports.index(client)
                tasks.append(self.connector(client_uri, serialized_data, client_index, websocket))
                st_count += 0
            await asyncio.gather(*tasks)
            print('3- Local model training at clients\' side finished')
            if compress:
                print('4- PS recovering and aggregating the received local models')
                for i in range(self.comp_len):
                    count = 0
                    for server_param in model.parameters():
                        if compress == 'quantize':
                            z_point = float(job_data['modelParam']['z_point'])
                            scale = float(job_data['modelParam']['scale'])
                            server_param.data += dequantize_tensor(self.q_diff[i][count], scale, z_point,
                                                                   self.info[i][count]) / len(self.q_diff)

                        count += 1

                global_weights = model.state_dict()

            else:
                print('4- PS aggregating received local models')
                # TODO local weights are not reset check it
                weights_avg = copy.deepcopy(self.local_weights[0])
                for k in weights_avg.keys():
                    for i in range(1, len(self.local_weights)):
                        weights_avg[k] += self.local_weights[i][k]

                    weights_avg[k] = torch.div(weights_avg[k], len(self.local_weights))

                global_weights = weights_avg
            # torch.save(model.state_dict(), "./ModelData/" + str(job_id) + '/model.pt')
            torch.save(model.state_dict(), 'model.pt')
            model.load_state_dict(global_weights)
            loss_avg = sum(self.local_loss) / len(self.local_loss)
            train_loss.append(loss_avg)
            g_loss, g_accuracy = self.testing(model, preprocessing, B_test, criterion)
            self.cumulative_size_uploaded_Mbytes_array.append(self.cumulative_size_uploaded_Mbytes)
            cumulative_energy = calculate_energy(self.cumulative_size_uploaded_Mbytes, data['avg_bitrate'])
            print('<<< Summary of round ' + str(curr_round) + ' >>>')
            print('Test accuracy: ', "{:.2f}".format(g_accuracy), '%')
            print('Test loss: ', "{:.2f}".format(g_loss))
            print('Cumulative size of transmitted data to the PS: ',
                  "{:.2f}".format(self.cumulative_size_uploaded_Mbytes), ' MB')
            print('Cumulative communication energy: ', "{:.2f}".format(cumulative_energy), ' Joules')
            print('<<< End summary of round ' + str(curr_round) + ' >>>')
            test_loss.append(g_loss)
            test_accuracy.append(g_accuracy)
            elapsed_time = round(time.time() - start_time, 2)
            if len(round_times) > 0:
                tot_time = round_times[-1] + elapsed_time
            else:
                tot_time = elapsed_time

            round_times.append(tot_time)
            if len(total_bytes) > 0:
                tot_bytes = total_bytes[-1] + self.bytes[-1] / 1e6
            else:
                tot_bytes = self.bytes[-1] / 1e6
            total_bytes.append(round(tot_bytes, 2))

            if curr_round == T:
                print('FL training completed. Final model saved at clients')
                total_energy = calculate_energy(self.cumulative_size_uploaded_Mbytes, data['avg_bitrate'])
                print(f'Total communication energy consumed during', T, 'rounds of training: ',
                      "{:.2f}".format(total_energy), ' Joules', f"with average bit rate: {data['avg_bitrate']:.2f}",
                      'bps')

                energy_per_round = calculate_energy(self.cumulative_size_uploaded_Mbytes_array[0], data['avg_bitrate'])
                '''
                if compress and compress == 'quantize':
                    model_size_bits = sum(p.numel() for p in model.parameters()) * int(job_data['modelParam']['num_bits'])
                    energy_per_round = calculate_energy_per_round(model_size_bits, data['avg_bitrate'])
                    total_energy = energy_per_round * T
                else:
                    model_size_bits = sum(p.numel() for p in model.parameters()) * 32         
                    energy_per_round = calculate_energy_per_round(model_size_bits, data['avg_bitrate'])
                    total_energy = energy_per_round * T
                '''
                serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round,
                                                            round_times,
                                                            total_bytes, True, energy_per_round=energy_per_round,
                                                            total_energy=total_energy)
            else:
                serialized_results = create_message_results(test_accuracy, train_loss, test_loss, curr_round,
                                                            round_times,
                                                            total_bytes, False)

            await websocket.send(serialized_results)
