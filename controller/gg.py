from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub

import switch
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

class SimpleMonitor13(switch.SimpleSwitch13):

    def __init__(self, *args, **kwargs):

        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.flow_model = None 
        self.scaler = MinMaxScaler()
        start = datetime.now()

        self.flow_training()

        end = datetime.now()
        print("Training time: ", (end-start))

    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)

            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):

        timestamp = datetime.now()
        timestamp = timestamp.timestamp()

        file0 = open("PredictFlowStatsfile.csv","w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        for stat in sorted([flow for flow in body if (flow.priority == 1) ], key=lambda flow:
            (flow.match['eth_type'],flow.match['ipv4_src'],flow.match['ipv4_dst'],flow.match['ip_proto'])):
        
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            if stat.match['ip_proto'] == 1:
                icmp_code = stat.match['icmpv4_code']
                icmp_type = stat.match['icmpv4_type']
                
            elif stat.match['ip_proto'] == 6:
                tp_src = stat.match['tcp_src']
                tp_dst = stat.match['tcp_dst']

            elif stat.match['ip_proto'] == 17:
                tp_src = stat.match['udp_src']
                tp_dst = stat.match['udp_dst']

            flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
          
            try:
                packet_count_per_second = stat.packet_count/stat.duration_sec
                packet_count_per_nsecond = stat.packet_count/stat.duration_nsec
            except:
                packet_count_per_second = 0
                packet_count_per_nsecond = 0
                
            try:
                byte_count_per_second = stat.byte_count/stat.duration_sec
                byte_count_per_nsecond = stat.byte_count/stat.duration_nsec
            except:
                byte_count_per_second = 0
                byte_count_per_nsecond = 0
                
            file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                        stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond))
            
        file0.close()

    def flow_training(self):

        self.logger.info("Flow Training ...")
        self.logger.info("Flow Training ...")
        flow_dataset = pd.read_csv('dataset.csv')
        self.logger.info("load...")

        flow_dataset.iloc[:, 5] = flow_dataset.iloc[:, 5].str.replace('.', '')
        flow_dataset.iloc[:, 3] = flow_dataset.iloc[:, 3].str.replace('.', '')
        flow_dataset=flow_dataset.drop(['timestamp','flow_id'],axis=1)
        self.logger.info("clean...")

        X_flow = flow_dataset.iloc[:, :-1].values
        y_flow = flow_dataset.iloc[:, -1].values
        y_flow = y_flow.astype(np.int32)
        X_flow = X_flow.astype(np.float32)

        X_flow_train, X_flow_test, y_flow_train, y_flow_test = train_test_split(X_flow, y_flow, test_size=0.25, random_state=0)
        
        X_flow_train1 = self.scaler.fit_transform(X_flow_train)
        X_flow_test1 = self.scaler.transform(X_flow_test)
        self.logger.info("pre...")

        
        class FlowModel(nn.Module):
             def __init__(self, input_size):
                 super(FlowModel, self).__init__()
                 self.fc1 = nn.Linear(input_size, 128)
                 self.dropout1 = nn.Dropout(0.5)
                 self.fc2 = nn.Linear(128, 64)
                 self.dropout2 = nn.Dropout(0.5)
                 self.fc3 = nn.Linear(64, 32)
                 self.dropout3 = nn.Dropout(0.5)
                 self.fc4 = nn.Linear(32, 1)

             def forward(self, x):
                 x = torch.sigmoid(self.fc1(x))
                 x = self.dropout1(x)
                 x = torch.sigmoid(self.fc2(x))
                 x = self.dropout2(x)
                 x = torch.sigmoid(self.fc3(x))
                 x = self.dropout3(x)
                 x = torch.sigmoid(self.fc4(x))
                 return x
        X_flow_train_torch = torch.tensor(X_flow_train1, dtype=torch.float32)
        y_flow_train_torch = torch.tensor(y_flow_train, dtype=torch.float32)
        X_flow_test_torch = torch.tensor(X_flow_test1, dtype=torch.float32)

# Training
        input_size = X_flow_train_torch.shape[1]
        self.flow_model = FlowModel(input_size)

# Define the optimizer and loss function
        optimizer = optim.Adam(self.flow_model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        epochs = 10
        batch_size = 32
        for epoch in range(epochs):
            print(epoch)
            for i in range(0, len(X_flow_train), batch_size):
                  inputs = X_flow_train_torch[i:i+batch_size]
                  targets = y_flow_train_torch[i:i+batch_size]

                  optimizer.zero_grad()
                  outputs = self.flow_model(inputs)
                  loss = criterion(outputs, targets.view(-1, 1))
                  loss.backward()
                  optimizer.step()

        with torch.no_grad():
               self.flow_model.eval()
               y_flow_pred_torch = self.flow_model(X_flow_test_torch)
        threshold = 0.5
        binary_predictions = torch.where(y_flow_pred_torch > threshold, 1, 0).flatten().numpy()

    # Calculate accuracy
        y_flow_test = y_flow_test.astype(int)
        acc = metrics.accuracy_score(y_flow_test, binary_predictions)
        
# Evaluation
        

        self.logger.info("succes accuracy = {0:.2f} %".format(acc*100))
        fail = 1.0 - acc
        self.logger.info("fail accuracy = {0:.2f} %".format(fail*100))
        self.logger.info("------------------------------------------------------------------------------")

    def flow_predict(self):
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            self.logger.info("read")
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset=predict_flow_dataset.drop(['timestamp','flow_id'],axis=1)
            self.logger.info("edited0")
            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype(np.float32)
            self.logger.info("edited1")
            X_predict_flow= self.scaler.transform(X_predict_flow)
            X_predict_flow_torch = torch.tensor(X_predict_flow, dtype=torch.float32)
            self.logger.info("loaded")
            with torch.no_grad():
               self.flow_model.eval()
               y_flow_pred_torch = self.flow_model(X_predict_flow_torch)
            threshold = 0.5
            binary_predictions = torch.where(y_flow_pred_torch > threshold, 1, 0).flatten().numpy()
            self.logger.info(binary_predictions)
            legitimate_trafic = 0
            ddos_trafic = 0

            for i in binary_predictions:
                if i == 0:
                    legitimate_trafic = legitimate_trafic + 1
                else:
                    ddos_trafic = ddos_trafic + 1
                    victim = int(predict_flow_dataset.iloc[i, 3])%20
                    
                    
                    

            self.logger.info("------------------------------------------------------------------------------")
            if (legitimate_trafic/len(binary_predictions)*100) >= 50:
                self.logger.info("legitimate trafic ...")
            else:
                self.logger.info("ddos trafic ...")
                self.logger.info("victim is host: h{}".format(victim))

            self.logger.info("------------------------------------------------------------------------------")
            
            file0 = open("PredictFlowStatsfile.csv","w")
            
            file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
            file0.close()

        except Exception as e:
               self.logger.error("Error occurred during file reading: {}".format(str(e)))
