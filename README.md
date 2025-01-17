# Federated Learning Platform

A distributed platform for executing federated learning tasks across multiple client nodes and a central server. This platform enables collaborative machine learning while keeping data localized on client hardware.

## System Architecture

### Components

1. **Parameter Server (PS)**
   - Central server that coordinates the federated learning process
   - Handles model distribution and aggregation
   - Manages connections with client nodes
   - Runs on port 8200 by default

2. **Client Nodes**
   - Process local data and perform model training
   - Communicate with PS via WebSocket connections
   

## Setup & Installation

### Prerequisites
- Docker
- Python 3.10


## Running the Platform

### Starting the Parameter Server
```bash
# Navigate to Server directory
cd Server

# Build the server image
docker build -t websocket-server .

# Run the server with number of expected clients (e.g., 3 clients)
docker run -p 8200:8200 websocket-server --num_ues 3
```

### Starting Client Nodes
```bash
# Navigate to Client directory
cd Client

# Build the client image
docker build -t websocket-client .

# Run  client instances on defined port
docker run -p 5000:5000 websocket-client 5000
docker run -p 5001:5001 websocket-client 5001

```

## Communication Protocol

The system uses WebSocket connections for bidirectional communication:
- Server endpoint: `/job_receive`
- Client endpoint: `/process`


### Federated Learning Workflow

1. **Task Initiation**
   - Send a WebSocket request to server endpoint `/job_receive` with a JSON payload as below. Change parameters accordingly.
   ```json
   {
     "jobData": {
       "general": {    
         "task": "satellite_fl",  // Task identifier (Any unique name)
         "algo": "regression",    // Type of algorithm. Supports regression/classification
         "host": "host_ip",       // Host IP address
         "clients": [             // List of clients connected to PS
        {
          "client_ip": "172.17.0.1:5000",  // Client IP address
          "client_id": "client1"           // Client name
        },
        {
          "client_ip": "172.17.0.1:5001",
          "client_id": "client2"
        },
        {
          "client_ip": "172.17.0.1:5002",
          "client_id": "client3"
        }
      ]
       },
       "scheme": {
         "minibatch": "64",      // Training minibatch size
         "epoch": "1",           // Number of local epochs
         "lr": "0.001",          // Learning rate for model training
         "scheduler": "random",  // Client scheduling method. Supports random/full/round_robin/latency
         "clientFraction": "0.7", // Client fraction to select number of participating client per communication round
         "minibatchtest": "4096", // Test minibatch size
         "comRounds": "20"        // Number of total communication rounds
       },
       "modelParam": {
         "optimizer": "Adam",  // Optimizer for training the model
         "loss": "Huber",      // Loss function used for training
         "compress": "quantize", // Model compression method. If no compression change to False
         "z_point": 0.0,       // Quantization parameter1
         "scale": 0.1,         // Quantization parameter2
         "num_bits": 16        // Quantization parameter3
       },
       "preprocessing": {
         "dtype": "regression", // data type. supports regression/img
         "folder": "satellite", // Folder where training data is stored. Should be inside data folder in the Clients
         "testfolder": "satellite", // Folder where test data is stores. Should be inside data  folder in Server.
         "normalize": false      // Set normalize to false
       }
     }
   }
   ```

2. **Client Selection**
   - PS selects a fraction of available clients based on `client_fraction`
   - Selected clients receive the initial model parameters

3. **Training Rounds**
   - For each communication round:
     1. PS sends model to selected clients via `/process` endpoint
     2. Clients perform local training using specified configuration
     3. Clients send updated models back to PS
     4. PS aggregates updates using FedAvg method. If quantized before sending model is dequantized before aggregating

4. **Termination**
   - Process continues until specified number of communication rounds completed
   - Final model is saved at PS

## Energy Consumption calculation

At the end of the training communication energy is calculated by calculating the **achievable bit rate** for a set of users in a wireless communication system. Below is an explanation of the key parameters, the steps in the calculation.

### Parameters

1. **Number of Users** (`num_users = 20`): 
2. **Area Size** (`A = 10000 m^2`): 
    - Defines a 100 x 100 m^2 area in which the users are randomly located.
3. **Transmission Power** (`Pt = 100e-3 W`):
4. **Bandwidth** (`B = 2e6` Hz):
    - The available bandwidth for the communication is 2 MHz.
5. **Noise Spectral Density** (`N0 = 1e-9` W/Hz):
6. **Model Size** (`32 * 1e6 bits`): 
    - Assumes a model with parameters represented using 32 bits.
7. **Server Position**:
    - Fixed at the center of the area (500, 500), ensuring a centralized setup for communication.

### Achievable Bit Rate Calculation

- For each user, the bit rate is calculated using the **Shannon-Hartley theorem**: 

  <img src="https://i.upmath.me/svg/R_i%20%3D%20B%20%5Ccdot%20%5Clog_2%5Cleft(1%20%2B%20%5Cfrac%7BP_t%7D%7Bd_i%5E2%20%5Ccdot%20B%20%5Ccdot%20N_0%7D%5Cright)" alt="R_i = B \cdot \log_2\left(1 + \frac{P_t}{d_i^2 \cdot B \cdot N_0}\right)" />

  Where:
    - <img src="https://i.upmath.me/svg/R_i" alt="R_i" />: Bit rate for user <img src="https://i.upmath.me/svg/i" alt="i" />.
    - <img src="https://i.upmath.me/svg/P_t" alt="P_t" />: Transmission power.
    - <img src="https://i.upmath.me/svg/d_i" alt="d_i" />: Distance to the server.
    - <img src="https://i.upmath.me/svg/B" alt="B" />: Bandwidth.
    - <img src="https://i.upmath.me/svg/N_0" alt="N_0" />: Noise spectral density.

Achievable bit rate is calculated at the start of PS based on the number of user equipments provided as the command line argument. This achievable bit rate is used to calculate energy consumption per communication round as follows.

* <img src="https://i.upmath.me/svg/%5Ctext%7BEnergy%20per%20round%7D%20%3D%20P_t%20%5Ctimes%20%5Ctext%7BTransmission%20Time%7D" alt="\text{Energy per round} = P_t \times \text{Transmission Time}" />


The total energy for the training is calculated as:

* <img src="https://i.upmath.me/svg/Total%20Energy%20%3D%20commRounds%20%5Ctimes%20Energy%20per%20round" alt="\text{Total Energy} = \text{commRounds} \times \text{Energy per round}" />


