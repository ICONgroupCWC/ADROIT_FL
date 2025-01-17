import json
import pickle
import base64



def create_message(batch_size, learning_rate, epochs, model, modelParam, transforms, weights=None):
    data = [batch_size, learning_rate, epochs, model, modelParam, transforms]
    if weights:
        data.append(weights)

    return pickle.dumps(data)


def create_message_results(accuracy, train_loss, test_loss, cur_round, elapsed_time, tot_bytes,  final_round=False,
                           weights=None, energy_per_round=0, total_energy=0):

    data = {'status': 'results', 'accuracy': str(accuracy[-1]), 'train_loss': str(train_loss[-1]),
            'test_loss': str(test_loss[-1]),
            "round": str(cur_round), "round_time": str(elapsed_time[-1]), 'total_bytes': str(tot_bytes[-1]), 'energy_per_round':float(energy_per_round), 'total_energy':float(total_energy)}

    if final_round:
        data['final'] = True
    else:
        data['final'] = False
    if weights:
        data['model'] = base64.b64encode(pickle.dumps(weights)).decode()

    serialized_data = json.dumps(data)

    return serialized_data


def create_result_dict(accuracy, train_loss, test_loss, cur_round, elapsed_time):
    data = {'accuracy': str(accuracy[-1]), 'train_loss': str(train_loss[-1]), 'test_loss': str(test_loss[-1]),
            "round": str(cur_round), "round_time": str(elapsed_time)}

    return data
