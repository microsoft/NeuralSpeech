#Reference: https://github.com/csteinmetz1/auraloss

import torch
import numpy as np
import librosa.filters

import scipy.signal


class SumAndDifference(torch.nn.Module):
    """Sum and difference signal extraction module."""

    def __init__(self):
        """Initialize sum and difference extraction module."""
        super(SumAndDifference, self).__init__()

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, #channels, #samples).
        Returns:
            Tensor: Sum signal.
            Tensor: Difference signal.
        """
        if not (x.size(1) == 2):  # inputs must be stereo
            raise ValueError(f"Input must be stereo: {x.size(1)} channel(s).")

        sum_sig = self.sum(x).unsqueeze(1)
        diff_sig = self.diff(x).unsqueeze(1)

        return sum_sig, diff_sig

    @staticmethod
    def sum(x):
        return x[:, 0, :] + x[:, 1, :]

    @staticmethod
    def diff(x):
        return x[:, 0, :] - x[:, 1, :]


class FIRFilter(torch.nn.Module):
    """FIR pre-emphasis filtering module.
    Args:
        filter_type (str): Shape of the desired FIR filter ("hp", "fd", "aw"). Default: "hp"
        coef (float): Coefficient value for the filter tap (only applicable for "hp" and "fd"). Default: 0.85
        ntaps (int): Number of FIR filter taps for constructing A-weighting filters. Default: 101
        plot (bool): Plot the magnitude respond of the filter. Default: False
    Based upon the perceptual loss pre-empahsis filters proposed by
    [Wright & Välimäki, 2019](https://arxiv.org/abs/1911.08922).
    A-weighting filter - "aw"
    First-order highpass - "hp"
    Folded differentiator - "fd"
    Note that the default coefficeint value of 0.85 is optimized for
    a sampling rate of 44.1 kHz, considering adjusting this value at differnt sampling rates.
    """

    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=101, plot=False):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps
        self.plot = plot

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            # convert analog filter to digital filter
            b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

            # compute the digital filter frequency response
            w_iir, h_iir = scipy.signal.freqz(b, a, worN=512, fs=fs)

            # then we fit to 101 tap FIR filter with least squares
            taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

            # now implement this digital FIR filter as a Conv1d layer
            self.fir = torch.nn.Conv1d(
                1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
            )
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

            if plot:
                from .plotting import compare_filters
                compare_filters(b, a, taps, fs=fs)

    def forward(self, input, target):
        """Calculate forward propagation.
        Args:
            input (Tensor): Predicted signal (B, #channels, #samples).
            target (Tensor): Groundtruth signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        input = torch.nn.functional.conv1d(
            input, self.fir.weight.data, padding=self.ntaps // 2
        )
        target = torch.nn.functional.conv1d(
            target, self.fir.weight.data, padding=self.ntaps // 2
        )
        return input, target

def apply_reduction(losses, reduction="none"):
    """Apply reduction to collection of losses."""
    if reduction == "mean":
        losses = losses.mean()
    elif reduction == "sum":
        losses = losses.sum()
    return losses

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module.
    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719).
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.
    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)
    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()
        self.log = log
        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(x_mag)
            y_mag = torch.log(y_mag)
        return self.distance(x_mag, y_mag)


class STFTLoss(torch.nn.Module):
    """STFT loss module.
    See [Yamamoto et al. 2019](https://arxiv.org/abs/1904.04472).
    Args:
        fft_size (int, optional): FFT size in samples. Default: 1024
        hop_size (int, optional): Hop size of the FFT in samples. Default: 256
        win_length (int, optional): Length of the FFT analysis window. Default: 1024
        window (str, optional): Window to apply before FFT, options include:
           ['hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of scaling frequency bins. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
        eps (float, optional): Small epsilon value for stablity. Default: 1e-8
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed.
            Default: 'mean'
        device (str, optional): Place the filterbanks on specified device. Default: None
    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'. By default.
        loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_bins=None,
        scale_invariance=False,
        eps=1e-8,
        output="loss",
        reduction="mean",
        device=None,
    ):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.device = device

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(log=True, reduction=reduction)
        self.linstft = STFTMagnitudeLoss(log=False, reduction=reduction)

        # setup mel filterbank
        if self.scale == "mel":
            assert sample_rate != None  # Must set sample rate to use mel scale
            assert n_bins <= fft_size  # Must be more FFT bins than Mel bins
            fb = librosa.filters.mel(sample_rate, fft_size, n_mels=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)
        elif self.scale == "chroma":
            assert sample_rate != None  # Must set sample rate to use chroma scale
            assert n_bins <= fft_size  # Must be more FFT bins than chroma bins
            fb = librosa.filters.chroma(sample_rate, fft_size, n_chroma=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

    def stft(self, x):
        """Perform STFT.
        Args:
            x (Tensor): Input signal tensor (B, T).
        Returns:
            Tensor: x_mag, x_phs
                Magnitude and phase spectra (B, fft_size // 2 + 1, frames).
        """
        x_stft = torch.stft(
            x,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            return_complex=True,
        )
        x_mag = torch.sqrt(
            torch.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=self.eps)
        )
        # x_phs = torch.angle(x_stft)
        return x_mag, x_stft

    def forward(self, x, y):
        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb, x_mag)
            y_mag = torch.matmul(self.fb, y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_mag_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        log_mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_mag_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0
        if self.w_phs:
            ignore_below = 0.1
            data = torch.cat([x_phs.real.unsqueeze(-1), x_phs.imag.unsqueeze(-1)], dim=-1).view(-1, 2)
            target = torch.cat([y_phs.real.unsqueeze(-1), y_phs.imag.unsqueeze(-1)], dim=-1).view(-1, 2)
            # ignore low energy components for numerical stability
            target_energy = torch.sum(torch.abs(target), dim=-1)
            pred_energy = torch.sum(torch.abs(data), dim=-1)
            target_mask = target_energy > ignore_below * torch.mean(target_energy)
            pred_mask = pred_energy > ignore_below * torch.mean(target_energy)
            indices = torch.nonzero(target_mask * pred_mask).view(-1)
            data, target = torch.index_select(data, 0, indices), torch.index_select(target, 0, indices)
            # compute actual phase loss in angular space
            data_angles, target_angles = torch.atan2(data[:, 0], data[:, 1]), torch.atan2(target[:, 0], target[:, 1])
            loss = torch.abs(data_angles - target_angles)
            # positive + negative values in left part of coordinate system cause angles > pi
            # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
            loss = np.pi - torch.abs(loss - np.pi)
            phs_loss = torch.mean(loss)
        else:
            phs_loss = 0.0

        # combine loss terms
        loss = (
            (self.w_sc * sc_mag_loss)
            + (self.w_log_mag * log_mag_loss)
            + (self.w_lin_mag * lin_mag_loss)
            + (self.w_phs * phs_loss)
        )

        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class MelSTFTLoss(STFTLoss):
    """Mel-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_mels=128,
        **kwargs,
    ):
        super(MelSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "mel",
            n_mels,
            **kwargs,
        )


class ChromaSTFTLoss(STFTLoss):
    """Chroma-scale STFT loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=256,
        win_length=1024,
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        n_chroma=12,
        **kwargs,
    ):
        super(ChromaSTFTLoss, self).__init__(
            fft_size,
            hop_size,
            win_length,
            window,
            w_sc,
            w_log_mag,
            w_lin_mag,
            w_phs,
            sample_rate,
            "chroma",
            n_chroma,
            **kwargs,
        )


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module.
    See [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480)
    Args:
        fft_sizes (list): List of FFT sizes.
        hop_sizes (list): List of hop sizes.
        win_lengths (list): List of window lengths.
        window (str, optional): Window to apply before FFT, options include:
            'hann_window', 'bartlett_window', 'blackman_window', 'hamming_window', 'kaiser_window']
            Default: 'hann_window'
        w_sc (float, optional): Weight of the spectral convergence loss term. Default: 1.0
        w_log_mag (float, optional): Weight of the log magnitude loss term. Default: 1.0
        w_lin_mag (float, optional): Weight of the linear magnitude loss term. Default: 0.0
        w_phs (float, optional): Weight of the spectral phase loss term. Default: 0.0
        sample_rate (int, optional): Sample rate. Required when scale = 'mel'. Default: None
        scale (str, optional): Optional frequency scaling method, options include:
            ['mel', 'chroma']
            Default: None
        n_bins (int, optional): Number of mel frequency bins. Required when scale = 'mel'. Default: None.
        scale_invariance (bool, optional): Perform an optimal scaling of the target. Default: False
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_bins=None,
        scale_invariance=False,
        **kwargs,
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [
                STFTLoss(
                    fs,
                    ss,
                    wl,
                    window,
                    w_sc,
                    w_log_mag,
                    w_lin_mag,
                    w_phs,
                    sample_rate,
                    scale,
                    n_bins,
                    scale_invariance,
                    **kwargs,
                )
            ]

    def forward(self, x, y):
        mrstft_loss = 0.0
        sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss = [], [], [], []

        for f in self.stft_losses:
            if f.output == "full":  # extract just first term
                tmp_loss = f(x, y)
                mrstft_loss += tmp_loss[0]
                sc_mag_loss.append(tmp_loss[1])
                log_mag_loss.append(tmp_loss[2])
                lin_mag_loss.append(tmp_loss[3])
                phs_loss.append(tmp_loss[4])
            else:
                mrstft_loss += f(x, y)

        mrstft_loss /= len(self.stft_losses)

        if f.output == "loss":
            return mrstft_loss
        else:
            return mrstft_loss, sc_mag_loss, log_mag_loss, lin_mag_loss, phs_loss


class RandomResolutionSTFTLoss(torch.nn.Module):
    """Random resolution STFT loss module.
    See [Steinmetz & Reiss, 2020](https://www.christiansteinmetz.com/s/DMRN15__auraloss__Audio_focused_loss_functions_in_PyTorch.pdf)
    Args:
        resolutions (int): Total number of STFT resolutions.
        min_fft_size (int): Smallest FFT size.
        max_fft_size (int): Largest FFT size.
        min_hop_size (int): Smallest hop size as porportion of window size.
        min_hop_size (int): Largest hop size as porportion of window size.
        window (str): Window function type.
        randomize_rate (int): Number of forwards before STFTs are randomized.
    """

    def __init__(
        self,
        resolutions=3,
        min_fft_size=16,
        max_fft_size=32768,
        min_hop_size=0.1,
        max_hop_size=1.0,
        windows=[
            "hann_window",
            "bartlett_window",
            "blackman_window",
            "hamming_window",
            "kaiser_window",
        ],
        w_sc=1.0,
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_phs=0.0,
        sample_rate=None,
        scale=None,
        n_mels=None,
        randomize_rate=1,
        **kwargs,
    ):
        super(RandomResolutionSTFTLoss, self).__init__()
        self.resolutions = resolutions
        self.min_fft_size = min_fft_size
        self.max_fft_size = max_fft_size
        self.min_hop_size = min_hop_size
        self.max_hop_size = max_hop_size
        self.windows = windows
        self.randomize_rate = randomize_rate
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_mels = n_mels

        self.nforwards = 0
        self.randomize_losses()  # init the losses

    def randomize_losses(self):
        # clear the existing STFT losses
        self.stft_losses = torch.nn.ModuleList()
        for n in range(self.resolutions):
            frame_size = 2 ** np.random.randint(
                np.log2(self.min_fft_size), np.log2(self.max_fft_size)
            )
            hop_size = int(
                frame_size
                * (
                    self.min_hop_size
                    + (np.random.rand() * (self.max_hop_size - self.min_hop_size))
                )
            )
            window_length = int(frame_size * np.random.choice([1.0, 0.5, 0.25]))
            window = np.random.choice(self.windows)
            self.stft_losses += [
                STFTLoss(
                    frame_size,
                    hop_size,
                    window_length,
                    window,
                    self.w_sc,
                    self.w_log_mag,
                    self.w_lin_mag,
                    self.w_phs,
                    self.sample_rate,
                    self.scale,
                    self.n_mels,
                )
            ]

    def forward(self, input, target):
        if input.size(-1) <= self.max_fft_size:
            raise ValueError(
                f"Input length ({input.size(-1)}) must be larger than largest FFT size ({self.max_fft_size})."
            )
        elif target.size(-1) <= self.max_fft_size:
            raise ValueError(
                f"Target length ({target.size(-1)}) must be larger than largest FFT size ({self.max_fft_size})."
            )

        if self.nforwards % self.randomize_rate == 0:
            self.randomize_losses()

        loss = 0.0
        for f in self.stft_losses:
            loss += f(input, target)
        loss /= len(self.stft_losses)

        self.nforwards += 1

        return loss


class SumAndDifferenceSTFTLoss(torch.nn.Module):
    """Sum and difference sttereo STFT loss module.
    See [Steinmetz et al., 2020](https://arxiv.org/abs/2010.10291)
    Args:
        fft_sizes (list, optional): List of FFT sizes.
        hop_sizes (list, optional): List of hop sizes.
        win_lengths (list, optional): List of window lengths.
        window (str, optional): Window function type.
        w_sum (float, optional): Weight of the sum loss component. Default: 1.0
        w_diff (float, optional): Weight of the difference loss component. Default: 1.0
        output (str, optional): Format of the loss returned.
            'loss' : Return only the raw, aggregate loss term.
            'full' : Return the raw loss, plus intermediate loss terms.
            Default: 'loss'
    Returns:
        loss:
            Aggreate loss term. Only returned if output='loss'.
        loss, sum_loss, diff_loss:
            Aggregate and intermediate loss terms. Only returned if output='full'.
    """

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        w_sum=1.0,
        w_diff=1.0,
        output="loss",
    ):
        super(SumAndDifferenceSTFTLoss, self).__init__()
        self.sd = SumAndDifference()
        self.w_sum = 1.0
        self.w_diff = 1.0
        self.output = output
        self.mrstft = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths, window)

    def forward(self, input, target):
        input_sum, input_diff = self.sd(input)
        target_sum, target_diff = self.sd(target)

        sum_loss = self.mrstft(input_sum, target_sum)
        diff_loss = self.mrstft(input_diff, target_diff)
        loss = ((self.w_sum * sum_loss) + (self.w_diff * diff_loss)) / 2

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sum_loss, diff_loss
