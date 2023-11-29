# Phase of recieved data.

$$
\begin{split}
\phi (t) &= \phi_{LO} - \phi_{RX} \\
&= 2\pi \int_{\tau_{LO}}^{t} (\omega_0 + \dot{\omega}(t' - \tau_{LO}))dt' + 2\pi \int_{\tau_{RX}}^{t} (\omega_0 + \dot{\omega}(t' - \tau_{RX}))dt' \\
&= 2\pi \{ \omega_0(t - \tau_{LO}) + \frac{1}{2} \dot{\omega}(t - \tau_{LO})^2 \} - 2\pi \{ \omega_0(t - \tau_{RX}) + \frac{1}{2} \dot{\omega}(t - \tau_{RX})^2 \} \\
&= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{(t^2 - 2t\tau_{LO} + \tau_{LO}^2) - (t^2 - 2t\tau_{RX} + \tau_{RX}^2)\} \} \\
&= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{2t(\tau_{RX} - \tau_{LO}) + \tau_{LO}^2 - \tau_{RX}^2 \} \}\\
&= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{2t(\tau_{RX} - \tau_{LO}) - (\tau_{RX} + \tau_{LO}) (\tau_{RX} - \tau_{LO}) \} \} \\
&= 2\pi \dot{\omega}(\tau_{RX} - \tau_{LO})t + 2\pi\{ \omega_0 (\tau_{RX} - \tau_{LO}) - \frac{1}{2} \dot{\omega} (\tau_{RX} + \tau_{LO}) (\tau_{RX} - \tau_{LO})\} \\
&= 2\pi \dot{\omega}(\tau_{RX} - \tau_{LO})t + \Delta \phi_0
\end{split}
$$
$\Delta \phi_0$は時刻$t$に無関係の項．つまり，周波数は$\dot{\omega}(\tau_{RX} - \tau_{LO})$として表せる．


#### あんまり関係なかったかも↓
$\tau = \tau_{RX} - \tau_{LO}$として，

$$
\begin{split}
\phi &= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{2t(\tau_{RX} - \tau_{LO}) - (\tau_{RX} + \tau_{LO}) (\tau_{RX} - \tau_{LO}) \} \} \\
&= 2\pi \omega_0 \tau + \pi \dot{\omega} \{2t \tau - \tau (\tau + 2 \tau_{LO}) \} \\
&= - \pi \dot{\omega} \tau^2  + 2\pi(\omega_0 + \dot{\omega}t - \dot{\omega} \tau_{LO}) \tau \\
\end{split} 
$$
従って，
$$
\begin{split}
&\tau^2 - 2( \frac{\omega_0}{\dot{\omega}} + t - \tau_{LO}) \tau + \frac{\phi}{\pi \dot{\omega}} = 0 \\
&\tau = \frac{\phi}{\pi \dot{\omega}} + \sqrt{ (\frac{\phi}{\pi \dot{\omega}})^2 + ( \frac{\omega_0}{\dot{\omega}} + t - \tau_{LO})}
\end{split} 
$$

# Route mean square velocity and vertical delay

相互相関関数 $F(V_{RMS}^2, \tau_{ver})$ から，２乗平均速度 $V_{RMS}^2$ とn番目の地下層までの鉛直方向遅れ時間 $\tau_{ver}$を求めることができる：

$$
F(V_{RMS}^2, \tau_{ver}) = \sum_{i \neq j} f_i \Big( \sqrt{\tau_{ver}^2 + \frac{L_i^2}{V_{RMS}^2}} \Big) \cdot f_j \Big( \sqrt{\tau_{ver}^2 + \frac{L_j^2}{V_{RMS}^2}} \Big)
$$

ここで，$f_i$ と $f_j$ は異なるそう受信点の組み合わせで得られたA-scan，$L_i, L_j$ は送受信点間の距離．

# Velocity in the n-th subsurface layer

n番目の地下層における電磁波速度：

$$
V_n = \sqrt{ \frac{\tau_n V_{RMS, n}^2 - \tau_{n-1} V_{RMS, n-1}^2}{\tau_n - \tau_{n-1}} }
$$


# Estimating $\Delta \tau$

FFTの周波数分解能（周波数軸の間隔）：
$$
\Delta f = \frac{f_s}{N} \ [\mathrm{Hz}]
$$
ただし，$f_s$はサンプリング周波数，$N$はデータ数．

遅れ時間$\tau$の分解能は，
$$
\Delta \tau = \frac{\Delta f}{\dot{\omega}} \ [\mathrm{s}]
$$

今回の実験の場合
- $f_s = 16 \ [\mathrm{kHz}]$
- $N = 32768 \ (=2^{15})$
- $\dot{\omega} = 0.9 \ [\mathrm{GHz/s}]$

より，
$$
\Delta \tau = \frac{16 \times 10^{3} / 2^{15}}{0.9 \times 10^9} \simeq 0.5425 \ [\mathrm{ns}]
$$
となる．


# Relation between $\Delta R$ and $\Delta \tau$
空間分解能$\Delta R$の見積り：
$$
\Delta R = \frac{c}{2 \Delta F \sqrt{\varepsilon_r}}
$$
ただし，$\Delta F$は周波数の変調幅である．
この式では前項目で登場したような，周波数分解能やデータ数は含まれていないが，前項目で導出した遅れ時間の分解能$\Delta \tau$の結果とは矛盾しない：
$$
\begin{split}
\Delta R &= \frac{v \Delta \tau}{2} \\
&= \frac{1}{2} \frac{f_z / N}{\dot{\omega}} \frac{c}{\sqrt{\varepsilon_r}} \\
\end{split}
$$
$f_s = N / T$（$T$はパルス長）より，
$$
\begin{split}
\Delta R &= \frac{1}{2} \frac{f_z / N}{\dot{\omega}} \frac{c}{\sqrt{\varepsilon_r}} \\
&= \frac{c}{2 N \dot{\omega} \sqrt{\varepsilon_r}} \\
&= \frac{c}{2 \Delta F \sqrt{\varepsilon_r}}
\end{split}
$$
が得られる．

# Estimating some representative $\tau$

TX5-RX1の場合
- 直達：$0.8 / 3 \times 10^8 = 2.666 \dots \ [\mathrm{ns}]$
- 床からの反射： $ 2 \sqrt{0.4^2 + 0.3^2} / (3 \times 10^8 / \sqrt{2}) \simeq 4.714 \ [\mathrm{ns}] $

TX5-RX2の場合
- 直達：$0.6 / 3 \times 10^8 = 2 \dots \ [\mathrm{ns}]$
- 床からの反射： $ 2 \sqrt{0.3^2 + 0.3^2} / (3 \times 10^8 / \sqrt{2}) = 4 \ [\mathrm{ns}] $

TX5-RX3の場合
- 直達：$0.4 / 3 \times 10^8 = 1.333 \dots \ [\mathrm{ns}]$
- 床からの反射： $ 2 \sqrt{0.2^2 + 0.3^2} / (3 \times 10^8 / \sqrt{2}) \simeq 3.399 \ [\mathrm{ns}] $

TX5-RX4の場合
- 直達：$0.2 / 3 \times 10^8 = 0.666 \dots \ [\mathrm{ns}]$
- 床からの反射： $ 2 \sqrt{0.1^2 + 0.3^2} / (3 \times 10^8 / \sqrt{2}) \simeq 2.981 \ [\mathrm{ns}] $