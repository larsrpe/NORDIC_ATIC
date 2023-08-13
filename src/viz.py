from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from functools import partial
import torch
import numpy as np
from typing import Callable


def viz_tracking(
    t,
    Y,
    L,
    pdf: Callable[[float, torch.Tensor], torch.Tensor],
    num_points: int,
    file_name="test",
):
    fig = plt.figure()

    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, L), ylim=(0, L))

    def animate(i):
        X = Y[:, i].reshape(-1, 2)
        x = X[:, 0]
        y = X[:, 1]

        t_i = t[i]
        xs = torch.linspace(0, L, num_points)
        ys = torch.linspace(0, L, num_points)
        z = torch.zeros(num_points, num_points)
        xx, yy = torch.meshgrid(xs, ys)

        X = torch.concatenate(
            (yy.flatten()[:, None], xx.flatten()[:, None]), axis=1
        ).double()
        z = pdf(t_i, X).reshape(num_points, num_points)
        z = z.numpy()[::-1, :]
        plt.clf()
        return plt.imshow(
            z, cmap="hot", interpolation="nearest", alpha=0.5, extent=[0, L, 0, L]
        ), plt.plot(x, y, "ko", markersize=5)

    anim = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=False)

    writer = FFMpegWriter(fps=60)
    anim.save(f"sims/tracking_{file_name}.mp4", writer=writer)


def viz_sim(t, y, L, file_name="test"):
    fig = plt.figure()

    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, L), ylim=(0, L))

    # initializing a line variable
    (plot,) = axis.plot([], [], "ko", markersize=5)

    # data which the line will
    # contain (x, y)
    def init():
        plot.set_data([], [])
        return (plot,)

    def animate(i, Y):
        X = Y[:, i].reshape(-1, 2)
        x = X[:, 0]
        y = X[:, 1]
        plot.set_data(x, y)
        return (plot,)

    anim = FuncAnimation(
        fig,
        partial(animate, Y=y),
        init_func=init,
        frames=len(t),
        interval=20,
        blit=True,
    )

    writer = FFMpegWriter(fps=60)
    anim.save(f"sims/{file_name}.mp4", writer=writer)


def viz_vectorfield(
    Field: Callable[[float, torch.Tensor], torch.Tensor],
    t_eval,
    L,
    num_points: int,
    file_name="test",
):
    fig = plt.figure()

    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, L), ylim=(0, L))

    # initializing a line variable

    # data which the line will
    # contain (x, y)

    def animate(i):
        t = t_eval[i]
        xs = np.linspace(0, L, num_points)
        ys = np.linspace(0, L, num_points)
        us = np.zeros(num_points**2)
        vs = np.zeros(num_points**2)
        xx = np.zeros(num_points**2)
        yy = np.zeros(num_points**2)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                u, v = tuple(Field(t, torch.tensor([x, y]).double()))
                us[i * num_points + j] = u.item()
                vs[i * num_points + j] = v.item()
                xx[i * num_points + j] = x
                yy[i * num_points + j] = y

        plt.clf()
        return plt.quiver(xx, yy, us, vs, scale=1)

    anim = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=False)

    writer = FFMpegWriter(fps=60)
    anim.save(f"sims/vectorfield_{file_name}.mp4", writer=writer)


def viz_sim_with_forces(
    t_eval, Field: Callable[[float, torch.Tensor], torch.Tensor], y, L, file_name="test"
):
    fig = plt.figure()

    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, L), ylim=(0, L))

    def animate(i, Y):
        X = Y[:, i].reshape(-1, 2)
        xs = X[:, 0]
        ys = X[:, 1]
        t = t_eval[i]
        us = np.zeros_like(xs)
        vs = np.zeros_like(ys)

        for i, x in enumerate(xs):
            y = ys[i]
            u, v = tuple(Field(t, torch.tensor([x, y]).double()))
            us[i] = u.item()
            vs[i] = v.item()

        plt.clf()
        return plt.plot(x, y, "ko", markersize=5), plt.quiver(xs, ys, us, vs, scale=1)

    anim = FuncAnimation(
        fig, partial(animate, Y=y), frames=len(t_eval), interval=20, blit=False
    )

    writer = FFMpegWriter(fps=60)
    anim.save(f"sims/sim_with_field_{file_name}.mp4", writer=writer)


def viz_pdf(
    pdf: Callable[[float, torch.Tensor], torch.Tensor],
    t_eval,
    L,
    num_points: int,
    file_name="test",
):
    fig = plt.figure()

    # marking the x-axis and y-axis
    axis = plt.axes(xlim=(0, L), ylim=(0, L))

    # initializing a line variable

    # data which the line will
    # contain (x, y)

    def animate(i):
        t = t_eval[i]
        xs = torch.linspace(0, L, num_points)
        ys = torch.linspace(0, L, num_points)
        xs = torch.linspace(0, L, num_points)
        ys = torch.linspace(0, L, num_points)
        z = torch.zeros(num_points, num_points)
        xx, yy = torch.meshgrid(xs, ys)

        X = torch.concatenate(
            (yy.flatten()[:, None], xx.flatten()[:, None]), axis=1
        ).double()
        z = pdf(t, X).reshape(num_points, num_points)
        z = z.numpy()[::-1, :]
        plt.clf()
        return plt.imshow(z)

    anim = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=False)

    writer = FFMpegWriter(fps=60)
    anim.save(f"sims/pdf_{file_name}.mp4", writer=writer)
