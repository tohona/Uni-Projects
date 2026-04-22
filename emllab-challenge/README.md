# emllab-challenge

Repository containing scripts and utilities for the EML-lab challenge.

## Using jupyter-lab

1. Set `PYTHONPATH`. Necessary in order to be able to import the `challenge`
module
    ```bash
    $ export PYTHONPATH=${PYTHONPATH}:$PWD
    ```
2. Run `jupyter-lab`
    ```bash
    $ jupyter-lab
    ```


## Using the scripts

The following commands are assumed to be executed in the root directory of
this repository.

1. Create virtual environment and install dependencies
    ```bash
    $ python -m venv venv
    $ echo $PWD > ./venv/lib/python3.11/site-packages/challenge.pth
    $ ./venv/bin/activate
    $ pip install -r requirements.txt
    ```

<!--
    $ echo "export PYTHONPATH=${PYTHONPATH}:$PWD" >> ./venv/bin/activate
-->

3. Execute the desired scripts. E.g.,
    ```bash
    $ python scripts/evaluate.py models/person_only_baseline/model_best.pt
    ```

## Examples

* Train person-only detection on the VOC-dataset
    ```bash
    $ python scripts/person_only_detection.py
    ```
* Evaluate a trained model using the VOC-dataset
    ```bash
    $ python scripts/evaluate.py <filename.pt>
    ```
* Evaluate the original `voc_pretrained`
    ```bash
    $ python scripts/evaluate.py models/voc_pretrained.pt --all-classes
    ```

## Installing COCO

```bash
sudo pacman -S aria2

mkdir data/COCO && cd data/COCO
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/train2017.zip
aria2c -x 10 -j 10 http://images.cocodataset.org/zips/val2017.zip
aria2c -x 10 -j 10 http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip *.zip
rm *.zip
```

