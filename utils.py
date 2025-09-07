import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import base64
import io
import os
from typing import Union
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.cluster import k_means
from sklearn.decomposition import PCA

from cnn import SmallBackbone, ClassifierHead, SmallCNN
from mlp import MLP

def embed_val_given_ckpt_path(
    ckpt_path: str = "log/lightning_logs/version_1/checkpoints/best-epoch=39-val_loss=0.0433.ckpt", 
    _model: str = 'cnn', 
    config_path: str = "config.yaml",
    dataset: str = 'val'):
    
    config = OmegaConf.load(config_path)
    
    if '_dim_' in ckpt_path:
        base_name = ckpt_path.split('.')[0].strip()
        emb_dim = int(base_name.split('_')[-1].strip())
    else:
        emb_dim = config.model.emb_dim
    
    # initialize model
    if _model == 'cnn':
        # print('Loading SmallCNN')
        backbone = SmallBackbone(
            num_channels_1=config.model.num_channels_1, 
            num_channels_2=config.model.num_channels_1, 
            emb_dim=emb_dim, 
            p=config.model.dropout)

        head = ClassifierHead(
            emb_dim=emb_dim, 
            num_classes=10, 
            p=config.model.dropout)

        model = SmallCNN.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            backbone=backbone,
            head=head)
        model.eval()
        
    elif _model == 'mlp':
        # print('Loading MLP')
        model = MLP.load_from_checkpoint(
        checkpoint_path=ckpt_path)
        model.eval()
        
    # load dataset
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if dataset in ['val', 'train']:
        full_train_val = datasets.MNIST("./data/", download=True, train=True, transform=transform)

        split = torch.load("data/MNIST/train_val_split.pt")
        train_idx, val_idx = split["train_idx"], split["val_idx"]

        val_dataset = Subset(full_train_val, val_idx)
        train_dataset = Subset(full_train_val, train_idx)
        
        if dataset == 'val':
            loader = DataLoader(
                val_dataset, 
                batch_size=config.data.batch_size, 
                num_workers=config.data.num_workers, 
                pin_memory=True, 
                persistent_workers=True)
        elif dataset == 'train':
            loader = DataLoader(
                train_dataset, 
                batch_size=config.data.batch_size, 
                num_workers=config.data.num_workers, 
                pin_memory=True, 
                persistent_workers=True)
    else: 
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_test = datasets.MNIST("./data/", download=True, train=False, transform=val_transform)

        loader = DataLoader(
            full_test, 
            batch_size=config.data.batch_size, 
            num_workers=config.data.num_workers, 
            pin_memory=True, 
            persistent_workers=True)

    # embedd images
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, labels = batch
            embeddings = model.encode(data)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.concat(all_embeddings)
    all_labels = torch.concat(all_labels)
    
    return all_embeddings, all_labels

def tsne_and_cluster_from_ckpt(    
    ckpt_dir: str = "log/lightning_logs/version_1/checkpoints/",
    ckpt_file: str = "best-epoch=39-val_loss=0.0433.ckpt", 
    model: str = 'cnn',
    config_path: str = "config.yaml"):
    
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    embeddings, labels = embed_val_given_ckpt_path(ckpt_path, model, config_path)
    
    if embeddings.shape[1] > 2:
        tsne = TSNE(
            n_components=2, 
            learning_rate='auto'
            ).fit_transform(embeddings)
    else:
        tsne = embeddings

    centroids, clusters, inertia = k_means(
        tsne, 
        n_clusters=10,
        n_init=5)
    
    k_means_acc, y_pred = cluster_majority_vote_accuracy(labels, clusters)    
    
    return embeddings, labels, tsne, clusters, k_means_acc

def cluster_majority_vote_accuracy(labels, clusters):
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    y = _to_numpy(labels).ravel()
    c = _to_numpy(clusters).ravel()

    mapping = {}
    for cid in np.unique(c):
        mask = c == cid
        if not np.any(mask):
            continue
        vals, counts = np.unique(y[mask], return_counts=True)
        mapping[cid] = vals[np.argmax(counts)]

    y_pred = np.array([mapping.get(cid, -1) for cid in c])
    acc = float((y_pred == y).mean())
    return acc, y_pred

def get_test_accuracy(
    ckpt_path: str, 
    _model: str, 
    config_path: str):

    config = OmegaConf.load(config_path)
    
    if '_dim_' in ckpt_path:
        base_name = ckpt_path.split('.')[0].strip()
        emb_dim = int(base_name.split('_')[-1].strip())
    else:
        emb_dim = config.model.emb_dim
    
    # Initialize model (same logic as embed_val_given_ckpt_path)
    if _model == 'cnn':
        backbone = SmallBackbone(
            num_channels_1=config.model.num_channels_1, 
            num_channels_2=config.model.num_channels_1, 
            emb_dim=emb_dim, 
            p=config.model.dropout)
        head = ClassifierHead(emb_dim=emb_dim, num_classes=10, p=config.model.dropout)
        model = SmallCNN.load_from_checkpoint(
            checkpoint_path=ckpt_path, backbone=backbone, head=head)
    elif _model == 'mlp':
        model = MLP.load_from_checkpoint(checkpoint_path=ckpt_path)
    
    model.eval()
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST("./data/", download=True, train=False, transform=transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True)
    
    # Compute accuracy
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

## Plotting functions

def plot_tsne(
    embeddings: Union[np.ndarray, torch.Tensor], 
    labels: Union[np.ndarray, torch.Tensor]):
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best", bbox_to_anchor=(1,1))
    plt.title("t-SNE")
    plt.show()

def interactive_tsne_over_checkpoints(
    ckpt_dir: str,
    ckpt_files: list[str],
    _model: str = "cnn",
    config_path: str = "config.yaml",
    point_size: int = 25,
    save_html: str | None = None,
):

    import numpy as np
    import pandas as pd
    import altair as alt
    
    dfs = []
    accuracies = {}
    kmeans_accuracies = {}

    for i, ckpt_file in enumerate(ckpt_files):
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        
        # test accuracy
        test_acc = get_test_accuracy(ckpt_path, _model, config_path)
        accuracies[i] = test_acc
        
        # extract embeddigs, cluster and tsne
        embeddings, labels, tsne, clusters, k_means_acc = tsne_and_cluster_from_ckpt(
            ckpt_dir=ckpt_dir,
            ckpt_file=ckpt_file,
            model=_model,
            config_path=config_path,
        )
        kmeans_accuracies[i] = k_means_acc 

        X = np.asarray(tsne)
        c = np.asarray(clusters).astype(int)
        y = np.asarray(labels).astype(int)

        df_i = pd.DataFrame({
            "x": X[:, 0],
            "y": X[:, 1],
            "cluster": c.astype(str),
            "label": y.astype(str),
            "ckpt_idx": i,
            "ckpt": ckpt_file,
            "test_accuracy": test_acc,
        })
        dfs.append(df_i)


    df_all = pd.concat(dfs, ignore_index=True)

    alt.data_transformers.disable_max_rows()

    # color toggle and checkpoint slider
    color_toggle = alt.param(
        name="color_by",
        value="cluster",
        bind=alt.binding_radio(options=["cluster", "label"], name="Color by: "),
    )

    ckpt_slider = alt.param(
        name="ckpt_idx",
        value=0,
        bind=alt.binding_range(min=0, max=len(ckpt_files) - 1, step=1, name="Checkpoint: "),
    )

    points = (
        alt.Chart(df_all)
        .transform_calculate(
            color="color_by == 'cluster' ? datum.cluster : datum.label"
        )
        .transform_filter("datum.ckpt_idx == ckpt_idx")
        .mark_point(filled=True, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="t-SNE 1")),
            y=alt.Y("y:Q", axis=alt.Axis(title="t-SNE 2")),
            color=alt.Color("color:N", legend=alt.Legend(title="Color")),
            tooltip=[
                # alt.Tooltip("ckpt:N", title="Checkpoint"),
                # alt.Tooltip("test_accuracy:Q", title="Test Accuracy", format=".2f"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("label:N", title="Label"),
                # alt.Tooltip("x:Q", format=".2f"),
                # alt.Tooltip("y:Q", format=".2f"),
            ],
            size=alt.value(point_size),
        )
        .add_params(color_toggle, ckpt_slider)
    )
    
    # small DataFrame for the accuracy
    acc_df = pd.DataFrame([
        {
            "ckpt_idx": i,
            "ckpt": ckpt_files[i],
            "test_text": f"Test acc: {accuracies[i]:.2f}",
            "kmeans_text": f"k-means acc: {kmeans_accuracies[i]:.2f}",
        }
        for i in accuracies.keys()
    ])
    
    accuracy_text = (
        alt.Chart(acc_df)
        .transform_filter("datum.ckpt_idx == ckpt_idx")
        .mark_text(align="right", baseline="top", dx=-10, dy=10, fontSize=14, fontWeight="bold", color="black")
        .encode(
            x=alt.value(690),
            y=alt.value(10),
            text=alt.Text("test_text:N"),
        )
    )

    kmeans_text = (
        alt.Chart(acc_df)
        .transform_filter("datum.ckpt_idx == ckpt_idx")
        .mark_text(align="right", baseline="top", dx=-10, dy=28, fontSize=14, fontWeight="bold", color="black")
        .encode(
            x=alt.value(690),
            y=alt.value(28),
            text=alt.Text("kmeans_text:N"),
        )
    )

    chart = points + accuracy_text + kmeans_text

    chart = chart.properties(
        width=700,
        height=600,
        title=f"t-SNE across checkpoints ({_model}) - Val embeddings, Test accuracy shown",
    ).interactive()

    if save_html:
        chart.save(save_html)

    return chart

def interactive_tsne_with_images(
    ckpt_path: str,
    _model: str = "cnn",
    config_path: str = "config.yaml",
    n_samples: int = 500,
    image_size: int = 14,
    point_size: int = 25,
    show_centroids: bool = True,
    save_html: str | None = None,
    random_seed: int = 42,
    overlay_scale: int = 8,
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    embeddings, labels = embed_val_given_ckpt_path(
        ckpt_path=ckpt_path, _model=_model, config_path=config_path, dataset="val"
    )

    # subsample
    N = len(embeddings)
    if N > n_samples:
        sel = np.random.choice(N, n_samples, replace=False)
        embeddings = embeddings[sel]
        labels = labels[sel]
        subsampled_idx = sel
    else:
        subsampled_idx = np.arange(N)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_val = datasets.MNIST("./data/", download=True, train=True, transform=transform)
    split = torch.load("data/MNIST/train_val_split.pt")
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    val_dataset = Subset(full_train_val, val_idx)

    imgs = []
    for i in subsampled_idx:
        img_tensor, _ = val_dataset[i]
        imgs.append(img_tensor)
    imgs = torch.stack(imgs, dim=0)

    def to_data_url(img_tensor, target_size=14):
        img = img_tensor.clone() * 0.3081 + 0.1307
        img = torch.clamp(img, 0, 1)
        arr = (img.squeeze().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L").resize((target_size, target_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    image_urls = [to_data_url(t, image_size) for t in imgs]

    E = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
    if E.ndim == 2 and E.shape[1] > 2:
        tsne = TSNE(n_components=2, learning_rate="auto", random_state=random_seed).fit_transform(E)
    else:
        tsne = E

    _, clusters, _ = k_means(tsne, n_clusters=10, n_init=5, random_state=random_seed)

    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    df = pd.DataFrame({
        "x": tsne[:, 0],
        "y": tsne[:, 1],
        "cluster": clusters.astype(str),
        "label": y.astype(str),
        "image_url": image_urls,
        "row_id": np.arange(len(tsne)),
    })

    test_acc = get_test_accuracy(ckpt_path, _model, config_path)

    alt.data_transformers.disable_max_rows()

    tsne_w, tsne_h = 700, 600
    panel_pad = 8
    panel_img_w = int(image_size * overlay_scale)
    panel_img_h = int(image_size * overlay_scale)
    panel_w = panel_img_w + 2 * panel_pad
    panel_h = panel_img_h + 2 * panel_pad

    # toggle color by cluster or label
    color_toggle = alt.param(
        name="color_by",
        value="cluster",
        bind=alt.binding_radio(options=["cluster", "label"], name="Color by: "),
    )

    hover_sel = alt.selection_point(
        name="hover",
        on="mouseover",
        fields=["row_id"],
        nearest=True,
        empty="none",
    )

    # tsne points panel
    points = (
        alt.Chart(df)
        .mark_point(filled=True, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="t-SNE 1")),
            y=alt.Y("y:Q", axis=alt.Axis(title="t-SNE 2")),
            color=alt.Color("color:N", legend=alt.Legend(title="Color")),
            tooltip=[
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("cluster:N", title="Cluster"),
            ],
        )
        .transform_calculate(
            color="color_by == 'cluster' ? datum.cluster : datum.label"
        )
        .add_params(color_toggle, hover_sel)
        .properties(width=tsne_w, height=tsne_h, title="t-SNE (val)")
    )

    # optional centroids
    layers_tsne = [points]
    if show_centroids:
        cent = (
            pd.DataFrame({"x": tsne[:, 0], "y": tsne[:, 1], "cluster": clusters})
            .groupby("cluster")
            .mean()
            .reset_index()
            .rename(columns={"x": "cx", "y": "cy"})
        )
        cent["cluster"] = cent["cluster"].astype(str)

        centroid_layer = (
            alt.Chart(cent)
            .transform_calculate(
                centroid_opacity="color_by == 'cluster' ? 1 : 0"
            )
            .mark_point(shape="cross", size=220, filled=False, stroke="black", strokeWidth=1.5)
            .encode(
                x="cx:Q",
                y="cy:Q",
                opacity=alt.Opacity("centroid_opacity:Q", legend=None),
                tooltip=[alt.Tooltip("cluster:N", title="Centroid (cluster)")],
            )
            .add_params(color_toggle)
        )
        layers_tsne.append(centroid_layer)

    # test accuracy
    accuracy_text = (
        alt.Chart(pd.DataFrame([{"test_accuracy": test_acc}]))
        .mark_text(
            align="right",
            baseline="top",
            dx=-10,
            dy=10,
            fontSize=14,
            fontWeight="bold",
            color="black",
        )
        .encode(
            x=alt.value(tsne_w),
            y=alt.value(0),
            text=alt.Text("test_accuracy:Q", format=".2f"),
        )
    )
    layers_tsne.append(accuracy_text)

    tsne_panel = alt.layer(*layers_tsne).properties(width=tsne_w, height=tsne_h).interactive()

    pos_x = panel_w / 2
    pos_y = panel_h / 2

    # separate image panel
    image_panel = (
        alt.Chart(df)
        .transform_filter(hover_sel)
        .mark_image(width=panel_img_w, height=panel_img_h)
        .encode(
            url=alt.Url("image_url:N"),
            x=alt.value(pos_x),
            y=alt.value(pos_y),
        )
        .add_params(hover_sel)
        .properties(width=panel_w, height=panel_h, title="Image")
    )

    chart = alt.hconcat(tsne_panel, image_panel).resolve_scale(color="independent")

    if save_html:
        chart.save(save_html)

    return chart

def interactive_tsne_over_PCAs(
    ckpt_dir: str,
    ckpt_file: str,
    principal_components: list[int], 
    _model: str = "cnn",
    config_path: str = "config.yaml",
    point_size: int = 25,
    save_html: str | None = None,
):

    dfs = []
    cum_explained = {}
    kmeans_acc_map = {}
    
    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    embeddings, labels = embed_val_given_ckpt_path(ckpt_path, _model, config_path)
    
    for i, pc in enumerate(principal_components):
        pca = PCA(n_components=pc)
        down_sampled_embeddings = pca.fit_transform(embeddings)

        if down_sampled_embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, learning_rate="auto").fit_transform(down_sampled_embeddings)
        else:
            tsne = down_sampled_embeddings

        centroids, clusters, inertia = k_means(tsne, n_clusters=10, n_init=5)
        
        km_acc, _ = cluster_majority_vote_accuracy(labels, clusters)
        kmeans_acc_map[i] = km_acc

        cum_explained[i] = float(np.sum(pca.explained_variance_ratio_))

        X = np.asarray(tsne)
        c = np.asarray(clusters).astype(int)
        y = np.asarray(labels).astype(int)

        df_i = pd.DataFrame({
            "x": X[:, 0],
            "y": X[:, 1],
            "cluster": c.astype(str),
            "label": y.astype(str),
            "pcs": pc,
            "pcs_index": i,
            "ckpt": ckpt_file,
        })
        dfs.append(df_i)

    df_all = pd.concat(dfs, ignore_index=True)

    alt.data_transformers.disable_max_rows()

    color_toggle = alt.param(
        name="color_by",
        value="cluster",
        bind=alt.binding_radio(options=["cluster", "label"], name="Color by: "),
    )

    pcs_index = alt.param(
        name="pcs_index",
        value=0,
        bind=alt.binding_range(min=0, max=len(principal_components) - 1, step=1, name="PCA k (index): "),
    )
    
    points = (
        alt.Chart(df_all)
        .transform_calculate(
            color="color_by == 'cluster' ? datum.cluster : datum.label"
        )
        .transform_filter("datum.pcs_index == pcs_index")
        .mark_point(filled=True, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="t-SNE 1")),
            y=alt.Y("y:Q", axis=alt.Axis(title="t-SNE 2")),
            color=alt.Color("color:N", legend=alt.Legend(title="Color")),
            tooltip=[
                alt.Tooltip("ckpt:N", title="Checkpoint"),
                alt.Tooltip("pcs:Q", title="PCA k"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("label:N", title="Label"),
            ],
            size=alt.value(point_size),
        )
        .add_params(color_toggle, pcs_index)
    )

    overlay_df = pd.DataFrame([
        {
            "pcs_index": i,
            "pcs": principal_components[i],
            "kmeans_text": f"k-means acc: {kmeans_acc_map[i]:.2f}",
            "ev_text": f"Expl. var: {cum_explained[i]*100:.1f}%",
        }
        for i in range(len(principal_components))
    ])

    kmeans_text = (
        alt.Chart(overlay_df)
        .transform_filter("datum.pcs_index == pcs_index")
        .mark_text(align="right", baseline="top", dx=-10, dy=28, fontSize=14, fontWeight="bold", color="black")
        .encode(x=alt.value(690), y=alt.value(28), text="kmeans_text:N")
        .add_params(pcs_index)
    )

    ev_text = (
        alt.Chart(overlay_df)
        .transform_filter("datum.pcs_index == pcs_index")
        .mark_text(align="right", baseline="top", dx=-10, dy=46, fontSize=14, fontWeight="bold", color="black")
        .encode(x=alt.value(690), y=alt.value(46), text="ev_text:N")
        .add_params(pcs_index)
    )

    chart = (points + kmeans_text + ev_text).properties(
        width=700,
        height=600,
        title=f"t-SNE with PCA downsampling ({_model})",
    ).interactive()

    if save_html:
        chart.save(save_html)

    return chart