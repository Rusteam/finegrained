# Fix classification labels

**Scenario**: we have a classification dataset with a few labels that are not
correct. We want to fix them. This classification dataset has been formed
from another detection dataset which contains original labels. That's why 
we want to find respective detection labels in the original dataset, fix
them and re-create our classification dataset.

We will use the following datasets:
- **classification_dataset** as a classification dataset, which
contains a sample tag with the name of the original dataset. It
also contains a label field **class_label** with classification labels.
- **detection_dataset** as a detection dataset. It contains
labels in a detection field **detection**.

**Steps**:
1. [Mark wrong labels in the classification dataset from the _FiftyOne App_.](#mark-wrong-labels)
2. Print the wrong labels in the classification dataset.
3. Assign a label tag to corresponding detection labels in the original dataset.
4. Send to the annotation server.
5. Annotate incorrect labels.
6. Fetch updated labels
7. Re-create the classification dataset without wrong labels.

## Solution

### Mark wrong labels.

Run `fiftyone app launch` and open classification_dataset in the _FiftyOne App_.
Select wrong labels and mark them with `fix_labels` sample tag.

### Print wrong labels

```bash
finegrained data display print_labels \
  --dataset classification_dataset \
  --label-field class_label \
  --include-tags fix_labels > fix_labels.txt
```
This will print all wrong labels in the `fix_labels.txt` file. 
Open the file and see its contents. It should contain wrong labels
selected in the previous step.

### Assign a label tag to detection labels

```bash
finegrained data tag tag_labels \
  --dataset detection_dataset \
  --label-field detection \
  --label-tags fix_labels \
  --labels fix_labels.txt
```

This will read labels from the file created at a step before
filter the detection dataset to only contain those labels
and assign a label tag to each detection label. Review results
in the FiftyOne App by navigating to **detection_dataset** and 
filtering with label tag **fix_labels**.

### Send to annotation server

```bash
finegrained data annotations annotate \
 --dataset detection_dataset \
 --annotation-key fix_errors \
 --label-field detection \
  --dataset-kwargs='{label_tags:"fix_labels"}' \
  --backend cvat_server.txt \
  --segment-size 10 \
  --project-id 1
```

This assumes that you have a CVAT server running and its 
connection parameters are defined in a `cvat_server.txt` file as following:
```
backend=cvat
url=http://<host>:<port>
username=<username>
password=<password>
```
It also expects a project with id `1` to exist in the CVAT server.

All samples containing labels with the `fix-dupl` tag will be sent to the CVAT server.

### Fix incorrect annotations

Navigate to your CVAT server and open an annotation task.
Use a filter on top-right corner to filter by your labels from `fix_labels.txt`
to avoid updating other correct labels.
Go through each task and fix incorrect labels.

Once finished, hit "Save" button and mark the task as finished.

### Fetch updated labels

```bash
finegrained annotations load 
    --dataset detection_dataset  
    --annotation-key fix_errors 
    --backend ./cvat_server.txt 
    --dest-field detection
``` 

This will fetch updated labels from the CVAT server and save them
to the `detection` field of the `detection_dataset` dataset.

If you open the dataset in the FiftyOne App, you will see that
the labels have been updated.
