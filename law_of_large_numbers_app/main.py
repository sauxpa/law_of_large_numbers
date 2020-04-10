import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.models import Band, ColorBar, ColumnDataSource, LinearColorMapper,\
    BasicTicker, Panel
from bokeh.models.widgets import Tabs, Slider
from bokeh.layouts import layout, WidgetBox
from bokeh.plotting import figure


def cov2corr(cov: np.ndarray) -> np.ndarray:
    D_inv = 1/np.sqrt(np.diag(cov))
    return np.multiply(D_inv, np.multiply(cov, D_inv))


def make_dataset(N,
                 alpha,
                 beta,
                 gamma,
                 ):
    """Creates a ColumnDataSource object with data to plot.
    """
    def rho(i: int, j: int) -> float:
        return alpha * (1 + np.abs(i-j)) ** (-beta) + gamma

    Cov = np.fromfunction(rho, (N, N), dtype=float)

    # Correlated sample
    sample = np.random.multivariate_normal(np.zeros(N), Cov)

    means = np.empty(N)
    stds = np.empty(N)

    n_samples = np.arange(1, N + 1)

    for i in range(N):
        means[i] = np.mean(sample[: i + 1])
        stds[i] = np.std(sample[: i + 1]) / np.sqrt(i + 1)

    df = pd.DataFrame(
        {
            'n_samples': n_samples,
            'means': means,
            'lower': means - stds,
            'upper': means + stds,
        }
    ).set_index('n_samples')

    Corr = cov2corr(Cov)

    dict_img = {
        'image': [Corr],
        }

    # Convert dataframe to column data source
    return ColumnDataSource(df), ColumnDataSource(data=dict_img)


def make_plot(src, src_img):
    """Create a figure object to host the plot.
    """
    fig = figure(
        plot_width=800,
        plot_height=400,
        title='Convergence',
        x_axis_label='n',
        y_axis_label='',
        )

    fig.line(
        'n_samples',
        'means',
        source=src,
        line_color='blue',
        alpha=0.8,
        legend='Empirical average',
        )

    fig.line(
        'n_samples',
        0.0,
        source=src,
        line_color='black',
        line_dash='dashed',
        alpha=0.5,
        legend='True mean',
        )

    band = Band(base='n_samples',
                lower='lower',
                upper='upper',
                source=src,
                level='underlay',
                fill_alpha=0.15,
                fill_color='red',
                line_width=1,
                line_color='black',
                )

    fig.add_layout(band)
    fig.legend.click_policy = 'hide'
    fig.legend.location = 'top_right'

    fig_img = figure(
        plot_width=500,
        plot_height=500,
        title='Correlation matrix',
        )

    color_mapper = LinearColorMapper(
        palette='Viridis256',
        low=0.0,
        high=1.0,
        )

    fig_img.image(image='image',
                  source=src_img,
                  x=0,
                  y=0,
                  dw=1,
                  dh=1,
                  color_mapper=color_mapper,
                  )

    fig_img.axis.visible = False

    color_bar = ColorBar(color_mapper=color_mapper,
                         ticker=BasicTicker(),
                         label_standoff=12,
                         border_line_color=None,
                         location=(0, 0),
                         )

    fig_img.add_layout(color_bar, 'right')

    return fig, fig_img


def update(attr, old, new):
    """Update ColumnDataSource object.
    """
    # Change parameters to selected values
    N = N_select.value
    alpha = alpha_select.value
    beta = beta_select.value
    gamma = gamma_select.value

    new_src, new_src_img = make_dataset(
        N,
        alpha,
        beta,
        gamma,
        )

    # Update the data on the plot
    src.data.update(new_src.data)
    src_img.data.update(new_src_img.data)


N_select = Slider(start=5,
                  end=2000,
                  step=1,
                  title='N',
                  value=100,
                  )

alpha_select = Slider(start=0.0,
                      end=1.0,
                      step=0.05,
                      title='alpha',
                      value=1.0,
                      )

beta_select = Slider(start=0.0,
                     end=3.0,
                     step=0.05,
                     title='beta',
                     value=1.0,
                     )

gamma_select = Slider(start=0.0,
                      end=1.0,
                      step=0.05,
                      title='gamma',
                      value=0.0,
                      )

# Update the plot when parameters are changed
N_select.on_change('value', update)
alpha_select.on_change('value', update)
beta_select.on_change('value', update)
gamma_select.on_change('value', update)

N = N_select.value
alpha = alpha_select.value
beta = beta_select.value
gamma = gamma_select.value

src, src_img = make_dataset(
    N,
    alpha,
    beta,
    gamma,
    )

controls = WidgetBox(
    N_select,
    alpha_select,
    beta_select,
    gamma_select,
    width=150,
    height=250,
    )

fig, fig_img = make_plot(src, src_img)

# Create a row layout
layout = layout(
    [
        [controls, fig_img],
        [fig]
    ],
)

# Make a tab with the layout
tab = Panel(child=layout, title='Law of large numbers')

tabs = Tabs(tabs=[tab])

curdoc().add_root(tabs)
