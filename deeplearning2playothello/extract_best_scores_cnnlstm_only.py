import os
import time
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data import CustomDatasetOne
from networks_2405536 import CNNLSTM


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def build_dev_loader(batch_size=256, len_samples=5, path_dataset="../dataset/"):
    """
    Build dev loader for CNN-LSTM (One2One with 5 frame sequences).
    """
    dataset_conf = {
        "filelist": "dev.txt",
        "len_samples": len_samples,
        "path_dataset": path_dataset,
        "batch_size": batch_size,
    }
    ds_dev = CustomDatasetOne(dataset_conf, load_data_once4all=True)
    return DataLoader(ds_dev, batch_size=batch_size)


def evaluate_models_in_dir(models_dir, dev_loader, device):
    best_acc = -1.0
    best_epoch = None
    best_file = None
    epochs_list = []
    accs_list = []

    if not os.path.isdir(models_dir):
        return None

    files = sorted([f for f in os.listdir(models_dir) if f.startswith('model_') and f.endswith('.pt')],
                   key=lambda x: int(x.split('_')[-1].split('.pt')[0]))
    for fname in files:
        path = os.path.join(models_dir, fname)
        try:
            model = torch.load(path, weights_only=False)
            try:
                model = model.to(device)
            except Exception:
                pass
            model.eval()
            rep = model.evalulate(dev_loader, device)
            acc_dev = rep.get("weighted avg", {}).get("recall", 0.0)
            if acc_dev is None:
                acc_dev = 0.0
            
            # extract epoch from filename
            try:
                epoch = int(fname.split('_')[-1].split('.pt')[0])
                epochs_list.append(epoch)
                accs_list.append(acc_dev)
            except Exception:
                pass
            
            if acc_dev > best_acc:
                best_acc = acc_dev
                best_file = fname
                best_epoch = epoch
        except Exception as e:
            # skip files that fail to load/evaluate
            continue

    if best_acc < 0:
        return None

    return {
        "best_epoch": best_epoch,
        "best_file": best_file,
        "best_acc": best_acc,
        "epochs": epochs_list,
        "accs": accs_list,
    }


def main():
    device = get_device()
    
    # Create logs folder if it doesn't exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    now = time.strftime('%Y-%m-%d %H:%M:%S')
    now_filename = time.strftime('%Y%m%d_%H%M%S')
    
    # Scan only save_models_CNN_LSTM* directories
    all_results = {}
    colors = {}
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    marker_list = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '+']
    
    cnnlstm_dirs = [item for item in sorted(os.listdir('.')) 
                    if item.startswith('save_models_') and ('CNN_LSTM' in item or 'CNNLSTM' in item) 
                    and os.path.isdir(item)]
    
    for i, item in enumerate(cnnlstm_dirs):
        try:
            dev_loader = build_dev_loader(len_samples=5)
            result = evaluate_models_in_dir(item, dev_loader, device)
            
            if result:
                all_results[item] = result
                colors[item] = {
                    'color': color_list[i % len(color_list)],
                    'marker': marker_list[i % len(marker_list)]
                }
        except Exception as e:
            print(f"Error processing {item}: {e}")
            continue
    
    # Write log with all results
    out_lines = [f"Log generated: {now}\n\n"]
    
    for model_dir in sorted(all_results.keys()):
        result = all_results[model_dir]
        out_lines.append(f"{model_dir}:\n")
        out_lines.append(f"  best_file: {result['best_file']}\n")
        out_lines.append(f"  best_epoch: {result['best_epoch']}\n")
        out_lines.append(f"  dev_recall: {round(100*result['best_acc'],4)}%\n\n")
    
    log_file = os.path.join(logs_dir, f'best_scores_cnnlstm_{now_filename}.log')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

    print(f'Wrote {log_file}')

    # Generate graph with CNN-LSTM models only
    plt.figure(figsize=(14, 8))
    
    for model_dir in sorted(all_results.keys()):
        result = all_results[model_dir]
        col_info = colors[model_dir]
        
        if result.get('epochs') and result.get('accs'):
            label = model_dir.replace('save_models_', '')
            plt.plot(result['epochs'], [100*acc for acc in result['accs']], 
                     marker=col_info['marker'], label=label, linewidth=2, 
                     color=col_info['color'], markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('CNN-LSTM Models - Accuracy vs Epoch (Dev Set)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    graph_file = os.path.join(logs_dir, f'accuracy_curves_cnnlstm_{now_filename}.png')
    plt.savefig(graph_file, dpi=100)
    print(f'Wrote {graph_file}')
    plt.close()


if __name__ == '__main__':
    main()
