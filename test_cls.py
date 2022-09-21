import torch
from torchvision import transforms
from PIL import Image
from torch.utils import data

from itertools import repeat
import pandas as pd
from tqdm import tqdm
from loguru import logger
import yaml
from datetime import datetime
import configargparse
import os
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib

import warnings

warnings.filterwarnings('ignore')


class SimpleDataset(torch.utils.data.Dataset):
    """Single Folder with images dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.images = list(Path(root_dir).glob(r'*.[jpb][pnm][jgp]'))
        self.images = [file for file in list(Path(root_dir).glob("*")) if
                       file.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.cpu().numpy())

        image = Image.open(str(self.images[idx]))

        if self.transform:
            image = self.transform(image)

        return image, 0, str(self.images[idx])


def get_metadata(path_to_folder_weights: str, path_to_test_data: str) -> dict[dict[str, int], str: str]:
    files = Path(path_to_folder_weights).glob("*")
    res_dict = {}

    for file in files:
        if file.name.startswith("efficient") or file.name.startswith("resne") or file.name.startswith("seresne") \
                or file.name.startswith("rexn"):
            res_dict["path_to_weights"] = str(Path(path_to_folder_weights) / file.name)

        elif file.stem == "mapping":
            mapping_path = Path(path_to_folder_weights) / 'mapping.yaml'

            with open(mapping_path, 'r') as f:
                animal2id = yaml.load(f, Loader=yaml.SafeLoader)
                id2animal = {v: k for k, v in animal2id.items()}
            res_dict["mapping"] = animal2id
            res_dict["mapping_network"] = id2animal

    # Compare the classes in the test folder with the classes that the classifier learned from
    gt_classes = [_cls.name for _cls in Path(path_to_test_data).glob("*")]
    pred_classes = list(res_dict["mapping"].keys())
    intersect_classes = set(gt_classes).intersection(set(pred_classes))
    classes_without_inference = set(gt_classes) - set(pred_classes)
    logger.warning(f"Классы, на которых сеть не обучалсь: {classes_without_inference}")

    res_dict["mapping"] = {x: res_dict["mapping"][x] for x in intersect_classes}

    # Create a table with true labels based on the classes the classifier was trained on
    table = None
    for cls_dir in tqdm(list(Path(path_to_test_data).glob("*"))):
        if cls_dir.name not in classes_without_inference:
            all_files = list(Path(cls_dir).glob("*"))
            dict_for_frame = {k.name: v for k, v in zip(all_files, repeat(cls_dir.name))}
            table_per_class = pd.DataFrame({"filename": dict_for_frame.keys(), "class": dict_for_frame.values()})
            table = pd.concat([table_per_class, table])

    # Delete filenames that are repeated more than once.
    names_more_2_times = (table["filename"].value_counts() > 1)[(table["filename"].value_counts() > 1)].index.to_list()
    logger.warning(
        f"Количество фотографий, которые повторяются и не участвуют в тестировании: {len(names_more_2_times)}")
    table = table[~table["filename"].isin(names_more_2_times)]

    def convert_from_class_to_label(class_name):
        return res_dict["mapping"][class_name]

    table["class_id"] = table.apply(lambda x: convert_from_class_to_label(x["class"]), axis=1)
    table = table.sort_values(by="filename")
    table = table.reset_index(drop=True)

    return table, res_dict


def inference(path_to_ims: str, res_dict: dict, input_size: int = 256, batch_size: int = 4) -> pd.DataFrame:
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    softmax = torch.nn.Softmax()

    path_to_w = res_dict["path_to_weights"]
    id2animal = res_dict["mapping_network"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SimpleDataset(path_to_ims, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size)

    model = torch.load(path_to_w)
    model.to(device)

    res = []

    with torch.set_grad_enabled(False):
        for _, batch in enumerate(dataloader):

            inputs, _, image_names = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            for i, out in enumerate(outputs):
                _, predicted_idx = torch.max(out, 0)
                predicted_idx = predicted_idx.cpu().item()
                probs_list = softmax(out).cpu().numpy()
                probs = probs_list[predicted_idx]
                res.append([Path(image_names[i]).name, predicted_idx, id2animal[
                    predicted_idx]])  # probs]) #{id2animal[k]:round(v,5) for k,v in enumerate(probs_list)}])

    df = pd.DataFrame(res, columns=['filename', 'class_id', 'class'])

    return df


def create_pred_table(path_to_test_data: str, res_dict: dict, input_size: int = 256,
                      batch_size: int = 4) -> pd.DataFrame:
    table = None
    for cls_dir in tqdm(list(Path(path_to_test_data).glob("*")), colour="green"):
        if cls_dir.stem in list(res_dict["mapping"].keys()):
            pred_table_per_cls = inference(path_to_ims=cls_dir, res_dict=res_dict, input_size=input_size,
                                           batch_size=batch_size)
            table = pd.concat([pred_table_per_cls, table])

    names_more_2_times = (table["filename"].value_counts() > 1)[(table["filename"].value_counts() > 1)].index.to_list()
    table = table[~table["filename"].isin(names_more_2_times)]
    table = table.sort_values(by="filename").reset_index(drop=True)
    return table


def count_metrics(GT_table: pd.DataFrame, Pred_table: pd.DataFrame, res_dict: dict):
    intersect_classes = set(GT_table["class"].values.tolist()).intersection(Pred_table["class"].values.tolist())

    report_metrics = classification_report(GT_table["class"].values, Pred_table["class"].values,
                                           target_names=sorted(Pred_table["class"].unique()),
                                           output_dict=True, digits=4)
    report_metrics = {k: report_metrics[k] for k in intersect_classes}

    # report_metrics_2 = classification_report(GT_table["class"].values, Pred_table["class"].values, target_names=sorted(Pred_table["class"].unique()), digits=3)

    # margin = len(Pred_table["class"].unique()) - len(GT_table["class"].unique())

    all_precisions = 0
    all_recall = 0
    all_f1_score = 0
    all_classes = list(report_metrics.keys())  # [:-(3+margin)]
    for class_anm in all_classes:
        all_precisions += report_metrics[class_anm]["precision"]
        all_recall += report_metrics[class_anm]["recall"]
        all_f1_score += report_metrics[class_anm]["f1-score"]
    print(f"Average precision: {all_precisions / len(all_classes) * 100:.3f}%")
    print(f"Average recall: {all_recall / len(all_classes) * 100:.3f}%")
    print(f"Average f1_score: {all_f1_score / len(all_classes) * 100:.3f}%")


def plot_cm(GT_table: pd.DataFrame, Pred_table: pd.DataFrame):
    cm = confusion_matrix(GT_table["class_id"].values, Pred_table["class_id"].values)
    figure, ax = plot_confusion_matrix(conf_mat=cm,
                                       class_names=sorted(Pred_table["class"].unique().tolist()),
                                       show_absolute=True,
                                       show_normed=True,
                                       colorbar=True,
                                       figsize=(27, 18),
                                       norm_colormap=matplotlib.colors.LogNorm())

    # time_path = datetime.now().strftime('%H:%M-%d-%m-%Y')

    time_path = datetime.now().strftime('%b-%d-%Y_%H:%M')
    # print(time_path)

    path_for_save = os.path.join("results_test", time_path)
    if not Path("results_test").exists():
        Path("results_test").mkdir()
    Path(path_for_save).mkdir()
    figure.savefig(os.path.join(path_for_save, "cm.jpg"))


def main(args):
    path_to_folder_weights = args.pt_w  # "/home/user/Lab/Vlad/Priroda/Article/weights/Classification/resnet101d"
    path_to_test_data = args.pt_data  # "/home/user/Lab/Vlad/Priroda/Datasets/16_test_train_val_cluster/test"

    GT_table, res_dict = get_metadata(path_to_folder_weights, path_to_test_data)

    Pred_table = create_pred_table(path_to_test_data, res_dict)

    count_metrics(GT_table, Pred_table, res_dict)
    plot_cm(GT_table, Pred_table)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--pt_w", "--path_to_folder_weights",
                        default="weights/Classification/tigers_vs_leopards/resnest101e", type=str,
                        help="Path to folder with .pt and mapping.yaml files")
    parser.add_argument("--pt_data", "--path_to_test_data", default="data/Classification/tigers_vs_leopards/test",
                        type=str, help="Path to folder with test data")

    args = parser.parse_args()

    main(args)
