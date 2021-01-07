# AI_final
## SSH for jupyter lab

- open remote to start jupyter lab server
    ```gherkin=
    ssh bruce@140.118.201.249
    jupyter lab --no-browser --port=8889
    ```
- copy token from server
- open jupyter lab from local
    ```gherkin=
    ssh -N -L localhost:8889:localhost:8889 bruce@140.118.201.249
    ```
## demo on webcam
- not available for remote(because of opencv) 
```gherkin=
python webcam_demo.py --model=<path_to_model>
```
## Train
```gherkin=
python train.py --result_dir=<path_to_work_dir>
```

## others
Please open other ipynb files:
1. demo_inclass.ipynb
- For printing out result from testing
2. draw_history.ipynb 
- For interactive training history
3. predicted_confusion.ipynb
- For drawing confusion matrix and calculate testing result
## utils
1. data_preprocess_new.py
- To get the attributes to train
