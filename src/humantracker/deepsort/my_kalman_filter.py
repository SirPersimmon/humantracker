import numpy as np
import scipy.linalg
from pykalman import KalmanFilter

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class MyKalmanFilter(object):
    def __init__(
        self,
        motion_mat=None,
        observation_mat=None,
        transition_covariance=None,
        observation_covariance=None,
    ):
        ndim, dt = 4, 1.0
        if motion_mat is None:
            self.motion_mat = np.eye(2 * ndim, 2 * ndim)
            for i in range(ndim):
                self.motion_mat[i, ndim + i] = dt
        else:
            self.motion_mat = motion_mat

        if observation_mat is None:
            self.observation_mat = np.eye(ndim, 2 * ndim)
        else:
            self.observation_mat = observation_mat

        self.kf = KalmanFilter(
            transition_matrices=self.motion_mat,
            observation_matrices=self.observation_mat,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        covariance = np.eye(8)
        return mean, covariance

    def predict(self, mean, covariance):
        return self.kf.filter_update(mean, covariance)

    def project(self, mean, covariance):
        mean = mean[:4]
        covariance = np.linalg.multi_dot(
            (self.observation_mat, covariance, self.observation_mat.T)
        )
        return mean.filled(), covariance

    def update(self, mean, covariance, measurement):
        mean, covariance = self.kf.filter_update(mean, covariance, measurement)
        return mean, covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
