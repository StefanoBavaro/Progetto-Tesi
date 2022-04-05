from manager import Manager

log_name="Production_Sorted"
activity_name = "Activity"
case_name = "Case ID"
timestamp_name = "Complete Timestamp"
outcome_name = 'label'
example_size = 4

manager = Manager(log_name, activity_name, case_name, timestamp_name, outcome_name, example_size)
manager.gen_internal_csv()

X_train, X_test, Y_train, Y_test, Z_train, Z_test = manager.csv_to_data()
#manager.build_neural_network_model(X_train,Y_train,Z_train)
manager.evaluate_model(X_test,Y_test,Z_test)





