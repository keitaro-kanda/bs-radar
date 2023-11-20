# Phase of recieved data.

$$
\begin{split}
\phi &= \phi_{LO} - \phi_{RX} \\
&= 2\pi \int_{\tau_{LO}}^{t} (\omega_0 + \dot{\omega}(t' - \tau_{LO}))dt' + 2\pi \int_{\tau_{RX}}^{t} (\omega_0 + \dot{\omega}(t' - \tau_{RX}))dt' \\
&= 2\pi \{ \omega_0(t - \tau_{LO}) + \frac{1}{2} \dot{\omega}(t - \tau_{LO})^2 \} - 2\pi \{ \omega_0(t - \tau_{RX}) + \frac{1}{2} \dot{\omega}(t - \tau_{RX})^2 \} \\
&= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{(t^2 - 2t\tau_{LO} + \tau_{LO}^2) - (t^2 - 2t\tau_{RX} + \tau_{RX}^2)\} \} \\
&= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{2t(\tau_{RX} - \tau_{LO}) + \tau_{LO}^2 - \tau_{RX}^2 \} \}\\
&= 2\pi \{ \omega_0(\tau_{RX} - \tau_{LO}) + \frac{1}{2}\dot{\omega} \{2t(\tau_{RX} - \tau_{LO}) - (\tau_{RX} + \tau_{LO}) (\tau_{RX} - \tau_{LO}) \} \}
\end{split}
$$

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
