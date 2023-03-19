import plotly.express as px
import pandas as pd
import plotly.io as pio   
pio.kaleido.scope.mathjax = None

df = pd.read_csv("cross-validation.csv")

df.category = df.category.str.replace("^covid","COVID-19 Positive", regex=True)
df.category = df.category.str.replace("non-covid","COVID-19 Negative")

def format_fig(fig):
    """Formats a plotly figure in a nicer way."""
    fig.update_layout(
        width=1200,
        height=550,
        plot_bgcolor="white",
        title_font_color="black",
        font=dict(
            family="Linux Libertine Display O",
            size=18,
            color="black",
        ),
    )
    gridcolor = "#dddddd"
    fig.update_xaxes(gridcolor=gridcolor)
    fig.update_yaxes(gridcolor=gridcolor)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')


fig = px.histogram(df, x="split", color="category", barmode="group", text_auto=True, log_y=True)
fig.update_layout(
    legend_title_text='Category',
    xaxis_title="Cross-Validation Fold",
    yaxis_title="Count",
)
format_fig(fig)
fig.write_image("cross-validation.png")
fig.write_image("cross-validation.pdf")
# path,split,has_covid,severity