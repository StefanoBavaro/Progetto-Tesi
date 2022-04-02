from manager import Manager

log_name="Production_Sorted"
activity_name = "Activity"
case_name = "Case ID"
timestamp_name = "Complete Timestamp"
outcome_name = 'label'
example_size = 4

manager = Manager(log_name, activity_name, case_name, timestamp_name, outcome_name, example_size)
manager.gen_internal_csv()
manager.csv_to_data()
manager.build_neural_network_model()




