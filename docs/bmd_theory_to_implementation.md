# BMD の理論と PyBMD 実装の対応

Schmidt (2020, *Nonlinear Dynamics*) の Bispectral Mode Decomposition (BMD) を、理論の各ステップが
PyBMD のどのコードに対応するかという観点で整理した文書です。式は PyBMD（および MATLAB 版
`refs/bmd/bmd.m`）が**実際に計算している形**で書いています。論文との表記の違いは §5.3 と §10 にまとめました。

実装全体の不変条件や MATLAB 版からの逸脱の詳細は [pybmd/bmd/CLAUDE.md](../pybmd/bmd/CLAUDE.md) を参照してください。

---

## 0. 全体の流れ

| 理論のステップ | 数式（概略） | 実装 |
| --- | --- | --- |
| 1. 変動成分をとる | $q' = q - \bar q$ | [base.py:513](../pybmd/bmd/base.py#L513) `select_mean`, [base.py:566](../pybmd/bmd/base.py#L566) |
| 2. ブロック分割（Welch 法） | $N_\mathrm{blk}$ 個の実現 $q^{[i]}$ | [base.py:399-406](../pybmd/bmd/base.py#L399-L406), [base.py:550](../pybmd/bmd/base.py#L550) `_get_block` |
| 3. 窓掛け＋時間 DFT | $\hat q^{[i]}_k$ | [base.py:557](../pybmd/bmd/base.py#L557) `_compute_blocks` |
| 4. トライアド $(k,l,k+l)$ の列挙 | $f_k+f_l=f_{k+l}$ | [utils.py:299](../pybmd/bmd/utils.py#L299) `triad_indices` |
| 5. 実現行列の組み立て | $\hat Q_{k+l},\ \hat Q_{k\circ l}=\hat Q_k\circ\hat Q_l$ | [standard.py:60](../pybmd/bmd/standard.py#L60) `_triad_matrices` |
| 6. バイスペクトル密度行列 | $\mathbf B = \hat Q_{k+l}^H \mathbf W \hat Q_{k\circ l}/N_\mathrm{blk}$ | [base.py:646](../pybmd/bmd/base.py#L646) |
| 7. 数値半径の最大化 | $\mathbf a_1=\arg\max_{\lVert\mathbf a\rVert=1}\lvert\mathbf a^H\mathbf B\mathbf a\rvert$ | [optimizers.py:105](../pybmd/bmd/optimizers.py#L105) `solve` |
| 8. モード・モードバイスペクトル | $\lambda_1,\ \phi_{k+l},\ \phi_{k\circ l}$ | [base.py:656-679](../pybmd/bmd/base.py#L656-L679) |

呼び出し順は `Base.fit()` → `_initialize` → `_compute_qhat` → `_triad_loop` → `_store_and_save` です
（[standard.py:24](../pybmd/bmd/standard.py#L24)）。

### 記号と変数名の対応

| 記号 | 意味 | 変数名 |
| --- | --- | --- |
| $N_t$ | スナップショット数 | `nt` |
| $N_\mathrm{fft}$ | 1 ブロックのスナップショット数 | `n_dft` |
| $N_\mathrm{ovlp}$ | ブロックの重なり（スナップショット数） | `n_overlap` |
| $N_\mathrm{blk}$ | ブロック数（＝実現の数） | `n_blocks` |
| $\Delta t$ | 時間刻み | `time_step`, `dt` |
| $n$ | 空間点数 × 変数数（1 モードの長さ） | `nxv = nx * nv` |
| $k, l$ | 符号付き周波数インデックス | `triads.k`, `triads.l`, `triads.kl` |
| $\hat Q_k$ | 周波数 $k$ の実現行列 $(n\times N_\mathrm{blk})$ | `q_hat[row]`, `q1`, `q2`, `q3` |
| $\mathbf W$ | 空間内積の重み（対角） | `weights`（形状 `(n, 1)`） |
| $\mathbf B$ | バイスペクトル密度行列 $(N_\mathrm{blk}\times N_\mathrm{blk})$ | `B` |
| $\mathbf a_1$ | 展開係数（最適化ベクトル） | `a`, `coeffs[i]` |
| $\lambda_1$ | 複素モードバイスペクトル | `r`, `L[f1_idx, f2_idx]` |
| $\phi_{k+l}$ | バイスペクトルモード | `modes[..., 0, ...]` |
| $\phi_{k\circ l}$ | 相互周波数場 (cross-frequency field) | `modes[..., 1, ...]` |

---

## 1. データと前処理

### 1.1 データの形

データは常に `(nt, *xshape, n_variables)`、つまり **時間が先頭、変数が末尾** です。
空間軸は NumPy/matplotlib の画像と同じ並びで、2 次元なら `xshape = (ny, nx)`、3 次元なら `(nz, ny, nx)`（x が最後の空間軸）です。
MATLAB や Fortran の配列は `(nx, ny)` で保存されているので、転置してから渡します（例: `u.transpose(0, 2, 1)`）。
内部では各スナップショットを C order で長さ $n = n_x n_v$ のベクトルに平坦化します
（[base.py:555](../pybmd/bmd/base.py#L555) の `reshape(self._n_dft, -1)`）。

### 1.2 平均の除去

BMD は変動成分 $q'(\mathbf x,t)=q(\mathbf x,t)-\bar q(\mathbf x)$ の相関を見る手法です。`mean_type` で選びます
（[base.py:513-543](../pybmd/bmd/base.py#L513-L543)）。

| `mean_type` | 引く量 | 備考 |
| --- | --- | --- |
| `'longtime'`（既定） | 全時間平均 $\bar q$ | MATLAB 版の既定と同じ |
| `'blockwise'` | 各ブロックの時間平均 | [base.py:568](../pybmd/bmd/base.py#L568) |
| `'zero'`, `'none'` | 何も引かない | 警告が出る |
| 引数 `mean=` | ユーザ指定の平均 | 形状 `(*xshape, nv)` 必須 |

### 1.3 空間内積の重み $\mathbf W$

2 つの場の内積を
$$
\langle \mathbf u, \mathbf v\rangle_{\mathbf W} = \mathbf u^H \mathbf W \mathbf v = \sum_j w_j\, \overline{u_j}\, v_j
$$
と定義します。$w_j$ は通常、求積（台形則など）の体積要素です。

- 生成: [pybmd/utils/weights.py](../pybmd/utils/weights.py) の `uniform`, `trapz_2d`, `trapz_3d`, `curvilinear_2d`（曲線格子）
- 形状チェック: [base.py:496](../pybmd/bmd/base.py#L496)。平坦ベクトルは受け付けません（並び順の曖昧さでモードが壊れるのを防ぐため）
- 平坦化: [base.py:424](../pybmd/bmd/base.py#L424)（データと同じ C order）

$\mathbf B$ は $\mathbf W$ に線形なので、重みの規約を変えると $|\lambda_1|$ 全体が定数倍されます
（[examples/example5_cylinder_paper.md](../examples/example5_cylinder_paper.md) の "Spatial weight" 節）。

---

## 2. スペクトル推定：ブロック分割・窓・DFT

### 2.1 ブロック分割

期待値 $E[\cdot]$ を、時系列を重なりのあるブロックに分けたアンサンブル平均で近似します（Welch 法）。

$$
N_\mathrm{blk} = \left\lfloor \frac{N_t - N_\mathrm{ovlp}}{N_\mathrm{fft} - N_\mathrm{ovlp}} \right\rfloor
$$

- 実装: [base.py:400-402](../pybmd/bmd/base.py#L400-L402)。$N_\mathrm{blk}<2$ ならエラーになります。
- `overlap` は**パーセント**（既定 50）、`n_overlap` は**スナップショット数**で、後者が優先されます
  （[base.py:359](../pybmd/bmd/base.py#L359)）。$N_\mathrm{ovlp}=\lfloor N_\mathrm{fft}\cdot\text{overlap}/100\rfloor$ です。
- $i$ 番目のブロックの開始位置は $\min(iN_\mathrm{step}+N_\mathrm{fft},N_t)-N_\mathrm{fft}$ です（$N_\mathrm{step}=N_\mathrm{fft}-N_\mathrm{ovlp}$）。
  最後のブロックがデータの末尾に揃えられます（[base.py:552](../pybmd/bmd/base.py#L552)）。

### 2.2 窓と DFT

ブロック $i$ の $j$ 番目のスナップショットに窓 $w_j$ を掛けて DFT をとります。

$$
\hat q^{[i]}_k = \frac{1}{\bar w\,N_\mathrm{fft}} \sum_{j=0}^{N_\mathrm{fft}-1} w_j\, q'^{[i]}_j\, e^{-2\pi \mathrm i\, jk/N_\mathrm{fft}},
\qquad \bar w = \frac{1}{N_\mathrm{fft}}\sum_j w_j
$$

- $1/\bar w$ は窓による振幅低下の補正です（`win_weight`、[base.py:428](../pybmd/bmd/base.py#L428)）。
  この規格化により、周波数 $f_k$ で振幅 $A$ の正弦波（窓がその周波数に合っている場合）は $|\hat q_k| = A/2$ になります。
- DFT と `fftshift`: [base.py:581-582](../pybmd/bmd/base.py#L581-L582)
- 窓: `'hamming'`（既定）/`'hann'`/`'boxcar'`/任意の配列（[utils.py:76](../pybmd/bmd/utils.py#L76)）

**両側スペクトルが必須です。** 差の相互作用（$l<0$）は負の周波数を使うので、`rfft` は使いません。

### 2.3 周波数軸

`fftshift` 後の行番号と符号付きインデックス $k$、物理周波数の関係は次のとおりです（[utils.py:105](../pybmd/bmd/utils.py#L105) `freq_axis`）。

$$
k \in \{-N_\mathrm{fft}/2, \dots, N_\mathrm{fft}/2-1\}\ (\text{偶数 } N_\mathrm{fft}),\qquad
f_k = \frac{k}{N_\mathrm{fft}\Delta t},\qquad k_\mathrm{Nyq} = N_\mathrm{fft}/2
$$

| 量 | 変数 |
| --- | --- |
| 符号付きインデックス $k$（行ごと） | `triads.f_idx` |
| 物理周波数 $f_k$ | `triads.freq`（`bmd.freq`） |
| Nyquist インデックス | `triads.f_nyq_idx` |
| $k$ → 行番号 | `triads.row_of(k)` |

`dt` を省略すると $\Delta t = 1/N_\mathrm{fft}$ とされ、$f_k = k$ になります（MATLAB 版と同じ）。

### 2.4 実現行列 $\hat Q_k$

周波数 $k$ の全ブロックの DFT を列に並べたものです。

$$
\hat Q_k = \begin{bmatrix} \hat q^{[1]}_k & \hat q^{[2]}_k & \cdots & \hat q^{[N_\mathrm{blk}]}_k \end{bmatrix} \in \mathbb C^{n\times N_\mathrm{blk}}
$$

実装では dict `q_hat[row]` に格納されます（[base.py:584](../pybmd/bmd/base.py#L584) `_compute_qhat`）。
メモリ節約のため、どれかのトライアドが参照する行（`triads.freq_needed`）だけを保持します。

---

## 3. トライアドと $f_1$–$f_2$ 平面

### 3.1 トライアド条件

2 次の非線形項（例：$\mathbf u\cdot\nabla\mathbf u$）は、周波数 $f_k$ と $f_l$ の成分の積から周波数 $f_k+f_l$ の成分を作ります。
このため BMD は、周波数の三つ組

$$
(f_k,\ f_l,\ f_{k+l}),\qquad f_k + f_l - f_{k+l} = 0
$$

（**トライアド**）を単位として解析します。インデックスで書けば $(k, l, k+l)$ です。
$l<0$ なら $f_{k+l} = f_k - |f_l|$ なので差の相互作用を表します。

### 3.2 計算するトライアドの選択

[utils.py:299](../pybmd/bmd/utils.py#L299) `triad_indices` は、次の条件をすべて満たす $(k,l)$ を列挙します（[utils.py:331-333](../pybmd/bmd/utils.py#L331-L333)）。

- $|k+l| < k_\mathrm{Nyq}$（和の周波数も軸上にあること）
- $|k|, |l| \le$ `max_freq_idx`（既定は Nyquist まで）
- $(k,l)$ が指定した `regions` のいずれかに入っていること

```
          f2 or l
             ^
     ________|
     |\      |\
     |  \  7 |  \
     | 6  \  | 8 /\
     |      \| / 1  \
 ----+-------+-------+-> f1 or k
      \  5 / |\      |
        \/ 4 |  \  2 |
          \  | 3  \  |
            \|______\|
```

各領域の判定式は [utils.py:126](../pybmd/bmd/utils.py#L126) `_region_masks` にあり、MATLAB 版と同一です。
`regions` は **1 始まり**（図のラベル番号）です。

### 3.3 既定値が `regions=[1, 2]` である理由（対称性）

$\mathbf B$ は次の 2 つの対称性を持ちます。

1. **$k \leftrightarrow l$ の入れ替え**：$\hat Q_k\circ\hat Q_l = \hat Q_l\circ\hat Q_k$ なので $\mathbf B(k,l)=\mathbf B(l,k)$。したがって $\lambda_1(k,l)=\lambda_1(l,k)$ です。
2. **実数データの共役対称性**：$\hat q_{-k} = \overline{\hat q_k}$ より $\mathbf B(-k,-l) = \overline{\mathbf B(k,l)}$。したがって $\lambda_1(-k,-l)=\overline{\lambda_1(k,l)}$ です。

この 2 つを使うと、実数データでは領域 1（和の相互作用 $k\ge l\ge 0$）と領域 2（差の相互作用 $k\ge|l|,\ l\le 0$）で平面全体を代表できます。
**複素数データ**では 2. が成り立たないので、必要に応じて他の領域も指定してください。

（乱数データで $L(3,2)=L(2,3)$ と $L(-3,-2)=\overline{L(3,2)}$ を数値的に確認済みです。）

### 3.4 `Triads` オブジェクト

| フィールド | 意味 |
| --- | --- |
| `f1_idx`, `f2_idx`, `f3_idx` | トライアド $i$ の $k,\ l,\ k+l$ の**行番号** |
| `k`, `l`, `kl` | 同じく**符号付きインデックス** |
| `f1`, `f2`, `f3` | 同じく物理周波数 |
| `region` | 所属領域（境界で重なる場合は番号の大きい方） |
| `triad_map` | `(n_freq, n_freq)` のトライアド番号表（対象外は −1） |
| `find(k, l)` | $(k,l)$ → トライアド番号 |

---

## 4. 古典バイスペクトルから BMD へ

### 4.1 古典バイスペクトル

1 点の信号 $q(t)$ のバイスペクトルは 3 次の相関

$$
S(f_k, f_l) = E\!\left[\hat q_k\, \hat q_l\, \overline{\hat q_{k+l}}\right]
$$

です。$f_k, f_l, f_{k+l}$ の位相が $\theta_k+\theta_l-\theta_{k+l}=\text{const}$ で結合している（**2 次の位相結合**）ときだけ、
ブロック平均で打ち消されずに大きな値が残ります。位相がランダムな成分は平均で消えます。

### 4.2 2 次項 $\hat q_{k\circ l}$

BMD では、周波数 $k$ と $l$ の成分の各点ごとの積（アダマール積）

$$
\hat q_{k\circ l}(\mathbf x) = \hat q_k(\mathbf x)\,\hat q_l(\mathbf x),\qquad \hat Q_{k\circ l} = \hat Q_k \circ \hat Q_l
$$

を 2 次の非線形項の代わりに使います。$\hat q_{k+l}$ と $\hat q_{k\circ l}$ の相関は、空間分布まで含めた古典バイスペクトルの一般化になっています。

- 実装: [standard.py:73-76](../pybmd/bmd/standard.py#L73-L76)（`q1 * q2` が $\hat Q_{k\circ l}$、`q3` が $\hat Q_{k+l}$）

### 4.3 バイスペクトル密度行列 $\mathbf B$

$$
\boxed{\ \mathbf B = \frac{1}{N_\mathrm{blk}}\, \hat Q_{k+l}^H\, \mathbf W\, \hat Q_{k\circ l}\ } \in \mathbb C^{N_\mathrm{blk}\times N_\mathrm{blk}},
\qquad
B_{ij} = \frac{1}{N_\mathrm{blk}} \left\langle \hat q^{[i]}_{k+l},\ \hat q^{[j]}_{k\circ l} \right\rangle_{\mathbf W}
$$

- 実装: [base.py:646](../pybmd/bmd/base.py#L646) `B = q_sum.conj().T @ (q_prod * weights) / self._n_blocks`
- MATLAB 版の `B = Q_hat_f3'*bsxfun(@times,Q_hat_f1.*Q_hat_f2,weight)/nBlks` と同じ式です（`'` は共役転置）。

**古典バイスペクトルとの関係.** $\mathbf B$ の対角成分は同じブロック内の相関、非対角成分は異なるブロック間の相関です。

$$
\operatorname{tr}\mathbf B = \frac{1}{N_\mathrm{blk}}\sum_i \left\langle \hat q^{[i]}_{k+l},\ \hat q^{[i]}_{k}\circ\hat q^{[i]}_{l}\right\rangle_{\mathbf W}
= \sum_j w_j\ \widehat{E}\!\left[\overline{\hat q_{k+l}}\,\hat q_k\hat q_l\right](\mathbf x_j)
$$

つまり $\operatorname{tr}\mathbf B$ は、各点の古典バイスペクトル $S(f_k,f_l)$ の推定値を空間積分したものです。
これは $\mathbf B$ の対角成分（同じブロック内の相関）だけを等しく足した量です。
BMD は非対角成分（異なるブロック間の相関）も含めて、ブロックの線形結合の係数 $\mathbf a$ を最適化します（次節）。

---

## 5. 最適化問題：数値半径

### 5.1 定式化

同じ係数 $\mathbf a\in\mathbb C^{N_\mathrm{blk}}$ で 2 つの場を実現の線形結合として作ります。

$$
\boldsymbol\psi_{k+l} = \hat Q_{k+l}\,\mathbf a,\qquad \boldsymbol\psi_{k\circ l} = \hat Q_{k\circ l}\,\mathbf a
$$

このとき

$$
\mathbf a^H \mathbf B\, \mathbf a = \frac{1}{N_\mathrm{blk}}\left\langle \boldsymbol\psi_{k+l},\ \boldsymbol\psi_{k\circ l}\right\rangle_{\mathbf W}
$$

です。この相関の大きさを最大にする $\mathbf a$ を求めます。

$$
\mathbf a_1 = \arg\max_{\lVert\mathbf a\rVert_2=1} \left|\mathbf a^H \mathbf B\,\mathbf a\right|,
\qquad
\lambda_1 = \mathbf a_1^H \mathbf B\,\mathbf a_1,
\qquad
|\lambda_1| = r(\mathbf B)
$$

$r(\mathbf B)$ は $\mathbf B$ の**数値半径**（field of values の最大絶対値）です。$\mathbf B$ はエルミートではないので、固有値問題ではなく数値半径の最大化問題になります。
制約 $\lVert\mathbf a\rVert_2=1$ は $\mathbf W$ を含まない通常のユークリッドノルムです。

- 実装: [base.py:652-655](../pybmd/bmd/base.py#L652-L655) `r, a = optimizers.mengi_overton(B, ...)`
- `r` は**複素数** $\lambda_1$ です（絶対値ではありません）。`L` にはこの複素数がそのまま入り、図にするとき $|L|$ をとります（[postproc.py:271](../pybmd/bmd/postproc.py#L271)）。

### 5.2 数値半径の性質とソルバ

回転させたエルミート部分

$$
\mathbf H(\theta) = \tfrac12\left(e^{\mathrm i\theta}\mathbf B + e^{-\mathrm i\theta}\mathbf B^H\right)
$$

を使うと、

$$
r(\mathbf B) = \max_{\theta\in[0,2\pi)} \lambda_{\max}\big(\mathbf H(\theta)\big)
$$

です。最大を与える $\theta^\ast$ での $\mathbf H(\theta^\ast)$ の最大固有ベクトルが $\mathbf a_1$ になります。

| 関数 | 理論上の役割 |
| --- | --- |
| [optimizers.py:24](../pybmd/bmd/optimizers.py#L24) `max_fov(A, theta)` | $\lambda_{\max}(\mathbf H(\theta))$ |
| [optimizers.py:58](../pybmd/bmd/optimizers.py#L58) `_dominant_eigvec(A, phi)` | $\mathbf H(\phi)$ の最大固有ベクトル $\mathbf a$ と $\mathbf a^H\mathbf B\mathbf a$ |
| [optimizers.py:105](../pybmd/bmd/optimizers.py#L105) `mengi_overton` | Mengi & Overton (2005) のレベルセット法。大域収束（唯一のソルバ） |
| [optimizers.py:70](../pybmd/bmd/optimizers.py#L70) `_pow2_scale` | $\lVert\mathbf B\rVert_1\in(1/2,1]$ への 2 のべき乗スケーリング |

**Mengi–Overton 法の要点.** レベル $w$ に対し、$\lambda_{\max}(\mathbf H(\theta)) = w$ となる角度 $\theta$ は、一般化固有値問題

$$
\mathbf R(w)\,\mathbf v = \mu\,\mathbf S\,\mathbf v,\qquad
\mathbf R(w)=\begin{bmatrix}2w\mathbf I & -\mathbf B^H\\ \mathbf I & \mathbf 0\end{bmatrix},\quad
\mathbf S=\begin{bmatrix}\mathbf B & \mathbf 0\\ \mathbf 0 & \mathbf I\end{bmatrix}
$$

の単位円上の固有値 $\mu = e^{\mathrm i\theta}$ として得られます（[optimizers.py:150-154](../pybmd/bmd/optimizers.py#L150-L154)）。
交差角で区切られた区間の中点のうち、$w$ を超えるものを次の候補にします。候補がなくなれば、現在のレベルが大域最大です。

**スケーリング.** 数値半径は $r(c\mathbf B)=c\,r(\mathbf B)$（$c>0$）を満たし、最大化ベクトルは変わりません。
実際の $\mathbf B$ は $1/N_\mathrm{blk}$ と重みのため非常に小さく（$\lVert\mathbf B\rVert_1\sim10^{-6}$ など）、MATLAB 版の絶対許容誤差では交差角が全て棄却されて過小評価が起きます。
PyBMD は 2 のべき乗で正規化してから解き（2 進浮動小数点で誤差なし）、最後に元の $\mathbf B$ で $\mathbf a^H\mathbf B\mathbf a$ を評価し直します。
この修正を含む MATLAB 版からの逸脱の詳細は [pybmd/bmd/CLAUDE.md](../pybmd/bmd/CLAUDE.md) の "Deviations" 節にあります。

### 5.3 論文表記との関係（$\mathbf B$ と $\mathbf B^H$）

論文本文は本リポジトリに含まれていないため、論文中の $\mathbf B$ の積の順序とは照合していません。
仮に論文が逆順 $\tilde{\mathbf B} = \hat Q_{k\circ l}^H\mathbf W\hat Q_{k+l}/N_\mathrm{blk} = \mathbf B^H$ で定義していても、

$$
\mathbf a^H\mathbf B^H\mathbf a = \overline{\mathbf a^H\mathbf B\,\mathbf a}
$$

なので、$|\lambda_1|$（モードバイスペクトル）と $\mathbf a_1$ は同じで、**複素数 $\lambda_1$ の位相だけが共役**になります。
PyBMD の `L` は MATLAB 版 `bmd.m` と同じ規約（上の $\mathbf B$）です。

### 5.4 $\mathbf a$ の位相の不定性

$\mathbf a\to e^{\mathrm i\alpha}\mathbf a$ としても $\mathbf a^H\mathbf B\mathbf a$ は変わりません。したがって $\lambda_1$ は一意ですが、
$\mathbf a_1$ とモードは**単位複素数倍の不定性**を持ちます。モード同士を比べるときは要素ごとではなく
$|\langle\mathbf u,\mathbf v\rangle|/(\lVert\mathbf u\rVert\lVert\mathbf v\rVert)\approx1$ で比べてください。
また、$\lambda_1$ が縮退していると（複数の $\mathbf a$ が同じ値を与えると）モードは一意に定まりません。

---

## 6. 出力される量

### 6.1 モードバイスペクトル $\lambda_1(f_k, f_l)$

トライアド $i$ の $\lambda_1$ は `L[f1_idx[i], f2_idx[i]]` に入ります（[base.py:658](../pybmd/bmd/base.py#L658)）。
`L` は `(n_freq, n_freq)` の複素配列で、計算しなかった $(k,l)$ は NaN です（[base.py:697-699](../pybmd/bmd/base.py#L697-L699)）。

- $|\lambda_1|$ が大きい ⇔ $(k,l,k+l)$ の間に空間的にコヒーレントな 2 次の位相結合がある
- 図: `plot_mode_bispectrum`（既定で $\log|\lambda_1|$）

### 6.2 モード

$$
\phi_{k+l} = \frac{\hat Q_{k+l}\mathbf a_1}{\lVert\hat Q_{k+l}\mathbf a_1\rVert_{\mathbf W}},\qquad
\phi_{k\circ l} = \frac{\hat Q_{k\circ l}\mathbf a_1}{\lVert\hat Q_{k\circ l}\mathbf a_1\rVert_{\mathbf W}},\qquad
\lVert\mathbf u\rVert_{\mathbf W}=\sqrt{\mathbf u^H\mathbf W\mathbf u}
$$

| モード | 名称 | インデックス | 実装 |
| --- | --- | --- | --- |
| $\phi_{k+l}$ | バイスペクトルモード（和の周波数の構造） | 0 | [base.py:656](../pybmd/bmd/base.py#L656), [base.py:667](../pybmd/bmd/base.py#L667) |
| $\phi_{k\circ l}$ | 相互周波数場（$k$ と $l$ の積が作る構造） | 1 | [base.py:657](../pybmd/bmd/base.py#L657), [base.py:668](../pybmd/bmd/base.py#L668) |
| $\phi_k$ | 構成モード（PyBMD 独自、`constituent_modes=True`） | 2 | [base.py:669-674](../pybmd/bmd/base.py#L669-L674) |
| $\phi_l$ | 同上 | 3 | 同上 |

- 正規化: [utils.py:363](../pybmd/bmd/utils.py#L363) `normalize_mode`
- 平坦ベクトル → 場の形 `(*xshape, nv)`: [base.py:190](../pybmd/bmd/base.py#L190) `_unflatten_modes`
- 取得: `bmd.get_modes_at_freqs(k, l)`、`bmd.get_modes_at_triad(i)`
- 図: `plot_triad_modes` は $\mathrm{Re}\,\phi$ と、相互作用の局在を示す $|\phi_{k\circ l}\cdot\phi_{k+l}|$（各点の積の絶対値）を描きます（[postproc.py:417](../pybmd/bmd/postproc.py#L417)）。

**注意**: $\phi_{k\circ l} \ne \phi_k\circ\phi_l$ です。$(\hat Q_k\circ\hat Q_l)\mathbf a \ne (\hat Q_k\mathbf a)\circ(\hat Q_l\mathbf a)$ だからです。
$\phi_k,\phi_l$ は同じ $\mathbf a_1$ で作った別の情報で、$\phi_{k\circ l}$ の分解ではありません。

### 6.3 エネルギー輸送項 $T$

$$
T(f_k,f_l) = \frac{1}{N_\mathrm{blk}}\,\mathrm{Re}\left[\left(\hat Q_{k+l}\mathbf a_1\right)^H\left(\hat Q_{k\circ l}\mathbf a_1\right)\right]
$$

- 実装: [base.py:662-663](../pybmd/bmd/base.py#L662-L663)。正規化前の $\boldsymbol\psi$ を使います。
- **重み $\mathbf W$ を含みません。** MATLAB 版と同じで、意図的です。
- したがって一様重み（$\mathbf W=\mathbf I$）なら $T = \mathrm{Re}\,\lambda_1$ です（数値的に確認済み）。
  重みがある場合は $\mathbf B$ から $\mathbf W$ を除いた Rayleigh 商の実部になります。
- 符号付きの量で、$f_{k+l}$ への（正）/からの（負）正味のエネルギー輸送を表します。図は `plot_energy_transfer`。

### 6.4 展開係数 $\mathbf a_1$

`coeffs` は `(n_triads, n_blocks)` で、各トライアドの $\mathbf a_1$ です（[base.py:664](../pybmd/bmd/base.py#L664)）。
モードは $\hat Q\,\mathbf a_1$ で再構成できるので、大規模な計算では `save_modes=False` として `coeffs.npy` だけを残せます
（ただし再構成用のヘルパーはまだありません）。

### 6.5 保存ファイル

`savedir/nfft{N}_novlp{M}_nblks{B}/` に保存されます（[base.py:752](../pybmd/bmd/base.py#L752) `_store_and_save`）。

| ファイル | 中身 |
| --- | --- |
| `bispectrum.npz` | `L`（$\lambda_1$）, `T`, `freq`, `f_idx` |
| `triads.npz` | `Triads` の全フィールド |
| `coeffs.npy` | $\mathbf a_1$ |
| `weights.npy` | 平坦化された $\mathbf W$ |
| `ltm_modes.npy` | 時間平均 $\bar q$ |
| `modes/triad_idx_XXXXXXXX.npy` | トライアドごとのモード `(n_comp, *xshape, nv)` |
| `params_modes.yaml` | パラメータ |

後処理（[pybmd/bmd/postproc.py](../pybmd/bmd/postproc.py)）はパスではなく fit 済みの `Standard`/`Cross` を受け取ります。保存しておく場合は `pickle` で `Standard` ごと保存し、読み込んでから渡します。

---

## 7. Cross-BMD（CBMD）

### 7.1 理論

実際の非線形項は、異なる変数の積の和です（例：$u\,\partial_x u + v\,\partial_y u$）。CBMD は、状態 $s$ と、それを作る積 $q\,r$ の和との相関を見ます。

$$
\hat Q_{k+l} \to \hat S_{k+l},\qquad
\hat Q_{k\circ l} \to \sum_{m} \hat Q^{(m)}_k \circ \hat R^{(m)}_l
$$

複数の状態 $s^{(1)},\dots,s^{(n_s)}$ を扱う場合は、それらを縦に積んで 1 本のベクトルにします。

$$
\hat Q_\mathrm{s} = \begin{bmatrix}\hat S^{(1)}_{k+l}\\ \vdots\\ \hat S^{(n_s)}_{k+l}\end{bmatrix},\qquad
\hat Q_\mathrm{qr} = \begin{bmatrix}\sum_m \hat Q^{(1,m)}_k\circ\hat R^{(1,m)}_l\\ \vdots\end{bmatrix},\qquad
\mathbf B = \frac{1}{N_\mathrm{blk}}\hat Q_\mathrm s^H\mathbf W\hat Q_\mathrm{qr}
$$

以降（数値半径、モード、$T$）は BMD と同じです。

### 7.2 実装

| 理論 | 実装 |
| --- | --- |
| $s$ のインデックス | `params['state_idx']`（**0 始まり**、既定 `[0]`） |
| $(q,r)$ の組 | `params['qr_idx']`（列が $q,r,q,r,\dots$ と交互、状態ごとに 1 行。既定 `[[1, 2]]`） |
| $\hat Q_\mathrm s$, $\hat Q_\mathrm{qr}$ の組み立て | [cross.py:153](../pybmd/bmd/cross.py#L153) `Cross._triad_matrices`（和は [cross.py:177-179](../pybmd/bmd/cross.py#L177-L179)） |
| 重み（空間のみ、状態数だけタイル） | [cross.py:128](../pybmd/bmd/cross.py#L128) |
| モードの形 `(*xshape, n_state)` | [cross.py:99](../pybmd/bmd/cross.py#L99) `_unflatten_modes` |

平坦軸は**状態が最も遅い添字**（`flat = j*nx + p`）です。そのため `_unflatten_modes` は `(n_state, *xshape)` に戻してから状態軸を末尾へ移します。
CBMD では `normalize_weights` と `constituent_modes` は使えません。

---

## 8. PyBMD 独自の拡張（MATLAB 版にないもの）

| 機能 | 理論上の意味 | 実装 |
| --- | --- | --- |
| `constituent_modes=True` | $\phi_k=\hat Q_k\mathbf a_1$, $\phi_l=\hat Q_l\mathbf a_1$ も出力 | [base.py:669-674](../pybmd/bmd/base.py#L669-L674) |
| `normalize_weights=True` | 変数ごとに $w\leftarrow w/\operatorname{var}(q_v)$（異なる単位の変数を揃える） | [weights.py:94](../pybmd/utils/weights.py#L94) |
| `normalize_data=True` | 各ブロック・各点・各変数を標準偏差で割る | [base.py:571-577](../pybmd/bmd/base.py#L571-L577) |
| `mean_type='blockwise'` | ブロックごとの平均を除去 | [base.py:568](../pybmd/bmd/base.py#L568) |
| `window='hann'/'boxcar'` | 窓の選択 | [utils.py:76](../pybmd/bmd/utils.py#L76) |
| MPI 並列 | トライアドをラウンドロビンで分配し `allreduce` | [base.py:638](../pybmd/bmd/base.py#L638), [base.py:690-694](../pybmd/bmd/base.py#L690-L694) |

`normalize_data=True` は複素数の入力データに対して分散の計算が誤っています（$|x|^2$ ではなく $x^2$ を使っている）。
実数データには影響しません。詳細は [docs/complex-conjugation-audit.md](complex-conjugation-audit.md)。

---

## 9. 最小の使用例と理論量の対応

```python
from pybmd.bmd.standard import Standard
import pybmd.utils.weights as W

params = dict(n_dft=64,            # N_fft
              time_step=dt,        # Δt
              n_space_dims=2, n_variables=1,
              overlap=50,          # N_ovlp = 32
              regions=[1, 2],      # 和と差の相互作用
              max_freq_idx=12)     # |k|,|l| ≤ 12
bmd = Standard(params, weights=W.trapz_2d(x, y, n_vars=1)).fit(data)  # data: (nt, ny, nx, 1)

i   = bmd.find_triad(12, 12)        # トライアド (12, 12, 24)
lam = bmd.L[bmd.triads.f1_idx[i], bmd.triads.f2_idx[i]]   # λ_1（複素数）
phi = bmd.get_modes_at_triad(i)     # phi[0] = φ_{k+l}, phi[1] = φ_{k∘l}
a1  = bmd.coeffs[i]                 # a_1
```

実例は [examples/example1_cylinder.py](../examples/example1_cylinder.py)（BMD）と [examples/example3_cbmd.py](../examples/example3_cbmd.py)（CBMD）を参照してください。

---

## 10. 読むときの注意点のまとめ

- `L` は**複素数** $\lambda_1$。モードバイスペクトルはその絶対値 $|\lambda_1|$。位相の規約は MATLAB 版と同じ（§5.3）。
- `T` は重みを含まない。一様重みなら $T=\mathrm{Re}\,\lambda_1$。
- モードは単位複素数倍の不定性を持つ（§5.4）。
- 実数データなら `regions=[1,2]` で平面全体を代表できる。複素数データでは不十分（§3.3）。
- `regions` は 1 始まり、`state_idx`/`qr_idx` は 0 始まり。
- $|\lambda_1|$ の絶対値は重み $\mathbf W$ の規約に比例して変わる。MATLAB 版の図と比べるときは一様重みを使う。
- ソルバ `MengiOverton` は MATLAB 版の過小評価を修正している。MATLAB 版との数値の差はこれが主因（[tests/octave/octave_cross_validation.md](../tests/octave/octave_cross_validation.md)）。
