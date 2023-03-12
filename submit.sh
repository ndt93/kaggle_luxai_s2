cp ./logs/exp_1/models/best_model.zip ./artifacts/saved_model_v1.zip
tar --exclude *.ipynb* --exclude __pycache__ --exclude *.gz -vczf submission.tar.gz *