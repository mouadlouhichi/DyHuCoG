import pandas as pd, matplotlib.pyplot as plt
import argparse, pathlib

def main(csv_path):
    df = pd.read_csv(csv_path)
    metrics = [c for c in df.columns if '@' in c]
    for m in metrics:
        plt.figure()
        plt.plot(df['epoch'], df[m])
        plt.xlabel('Epoch')
        plt.ylabel(m)
        plt.title(m + ' over training')
        out_path = pathlib.Path(csv_path).parent / f"{m}.png"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    print('Saved plots to', pathlib.Path(csv_path).parent)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', help='metrics.csv path')
    args = ap.parse_args()
    main(args.csv)
