import plotly.express as px
import pandas as pd

df = pd.read_csv("cross-validation.csv")

fig = px.histogram(df, x="split", color="category", barmode="group", text_auto=True)
fig.write_image("cross-validation.png")

# path,split,has_covid,severity