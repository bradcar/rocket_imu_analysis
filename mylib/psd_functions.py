# psd_functions.py
"""
POWER SPECTRAL DENSITY FUNCTIONS
"""
import matplotlib.pyplot as plt
from scipy import signal


def get_psd(signal_vector, fs, nperseg=1024):
    """
    Power spectral density (PSD) calculation

    Usage:
        ax_freq, ax_psd = get_psd(ax_b, fs=sample_frequency, nperseg=1024)
        plot_psd(ax_freq, ax_psd, title="PSD of Ax_b (whole input flight_sensor_input_data)")
        print(f"{len(ax_b)=}, {len(ax_psd)=}")

    :param signal_vector: input
    :param fs: sampling frequency
    :param nperseg:  number of points for PSD
    :return:
    """
    nperseg = min(nperseg, len(signal_vector))

    f, psd = signal.welch(
        signal_vector,
        fs=fs,
        nperseg=nperseg,
        window="hann",
        detrend="constant",
        scaling="density"
    )

    return f, psd


def plot_psd(f, psd, title="Power Spectral Density"):
    plt.figure(figsize=(8, 4))
    plt.loglog(f[1:], psd[1:])  # skip DC bin
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (m²/s⁴/Hz)")
    plt.title(title)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def rms_from_psd(psd, df):
    """
    usage:
    f, psd = get_psd(ax_b, fs=sample_frequency)
    df = f[1] - f[0]  # frequency spacing
    rms_cum = rms_from_psd_np(psd[1:], df) # skip DC bin (psd[0])

    df_rms = rms_from_psd(psd)
    fig = fig_from_df(df_rms)
    fig.update_xaxes(type="log", title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Acceleration gRMS")
    fig.write_html('cum-rms.html', full_html=False, include_plotlyjs='cdn')
    df_rms.to_csv('cum-rms.csv')
    fig.show()
    """
    # Multiply PSD by frequency bin width
    psd_scaled = psd * df
    psd_cumsum = np.cumsum(psd_scaled)
    rms_cum = np.sqrt(psd_cumsum)
    return rms_cum
