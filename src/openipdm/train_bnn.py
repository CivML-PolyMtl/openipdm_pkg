from ssm_kr import SSM_KR
import os

det_model = SSM_KR()
file_dir = os.path.dirname(__file__)
data_path = os.path.join(file_dir, 'data', 'infr_det_syn_data.csv')
in_dim = 1
out_dim = 1
det_model.BNN.load_csv_data(data_path, in_dim, out_dim)
det_model.BNN.train()
det_model.BNN.task.predict_test_set(std_factor=2)
