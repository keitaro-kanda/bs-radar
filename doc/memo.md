#Phase of recieved data.

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
