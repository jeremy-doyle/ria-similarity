# -*- coding: utf-8 -*-

from dash import Dash, html, dcc, dash_table, Input, Output
import dash.dash_table.FormatTemplate as FormatTemplate
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sqlite3 as db

db_location = "data/advisor_similarity.db"


def get_names(region):
    con = db.connect(db_location)
    query = f"""
        select * from ID_lookup_{region}
    """
    lookup = pd.read_sql(query, con)
    con.close()
    return lookup["Name"].tolist()


def get_ID(region, name):
    con = db.connect(db_location)
    query = f"""
        select * from ID_lookup_{region} where Name = '{name}'
    """
    lookup = pd.read_sql(query, con)
    con.close()
    return lookup["ID"].values[0]


def get_neighbors(region, id_, k):
    con = db.connect(db_location)
    query = f"""
        select * from neighbors_{region} 
        where ID = '{id_}' and neighbor < {k + 1}
    """
    neighbors = pd.read_sql(query, con)
    con.close()
    return neighbors


def get_neighbors_datatable(region, id_, k):
    neighbors = get_neighbors(region, id_, k)
    where_IDs = [f"'{id_}'" for id_ in neighbors["neighbor_ID"]]
    where_IDs = ",".join(where_IDs)
    con = db.connect(db_location)
    query = f"""
        select * from reporting_data_formatted
        where ID in ({where_IDs})
    """
    data = pd.read_sql(query, con)
    con.close()
    df = neighbors.merge(data, left_on="neighbor_ID", right_on="ID")
    df = df.iloc[:, 4:]
    return df


def main(region, advisor, k):
    id_ = get_ID(region, advisor)
    df = get_neighbors_datatable(region, id_, k)
    return df


xaxis_type = {
    "Discretionary Assets": "log",
    "Non-Discretionary Assets": "log",
    "Total Assets": "log",
    "Discretionary Accounts": "log",
    "Non-Discretionary Accounts": "log",
    "Total Accounts": "log",
    "Advisory Employees": "log",
    "Investment Advisor Reps": "log",
    "Broker-Dealer Reps": "log",
    "Dually-Registered IARs": "log",
    "Insurance Agents": "log",
    "Outside Solicitors": "log",
    "Assets Per Advisor": "log",
    "Clients Per Advisor": "log",
    "Assets Per Client": "log",
    "Discretionary to Total Assets": "linear",
    "Advisor Growth (1y)": "linear",
    "Advisor Growth (3y)": "linear",
    "Assets Growth (1y)": "linear",
    "Assets Growth (3y)": "linear",
    "Assets Per Advisor Growth (1y)": "linear",
    "Assets Per Advisor Growth (3y)": "linear",
    "Clients Per Advisor Growth (1y)": "linear",
    "Clients Per Advisor Growth (3y)": "linear",
    "Assets Per Client Growth (1y)": "linear",
    "Assets Per Client Growth (3y)": "linear",
    "Discretionary to Total Assets Growth (1y)": "linear",
    "Discretionary to Total Assets Growth (3y)": "linear",
}


def create_blank_fig():
    fig = go.Figure()
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, plot_bgcolor="#fff")
    return fig


def create_ecdf(region, advisor, k, column):
    if advisor == "" or advisor == None or column == None:
        return create_blank_fig()
    else:
        # filter by region
        con = db.connect(db_location)
        region_filter = pd.read_sql(f"select * from ID_lookup_{region}", con)
        data = pd.read_sql(f'select ID, "{column}" from reporting_data_formatted', con)
        con.close()
        data = region_filter.merge(data, on="ID")

        # identify neighbors to advisor
        id_ = get_ID(region, advisor)
        neighbors = get_neighbors(region, id_, k)
        neighbors = neighbors.loc[:, ["neighbor_ID", "neighbor"]]
        neighbors = neighbors.rename(columns={"neighbor_ID": "ID"})
        neighbors["neighbor"] = True
        data = data.merge(neighbors, how="left", on="ID")
        data["neighbor"] = data["neighbor"].fillna(False)

        # drop na and inf records in column
        data = data.loc[~data[column].isin([np.nan, np.inf, -np.inf]), :]

        # set plot colors
        blue = "#1EAEDB"
        red = "#DB4B1E"
        dark_blue = "#157998"

        # create ecdf plot (remove hover labels)
        fig = px.ecdf(
            data,
            x=column,
            markers=True,
            lines=False,
            ecdfnorm="percent",
            opacity=0.7,
            hover_name="Name",
            color_discrete_sequence=[blue],
        )
        fig.update_traces(hovertemplate=None, hoverinfo="skip")

        # get the points from the ecdf plot
        fig_data = fig.data[0]
        fig_data = pd.DataFrame(
            {"Name": fig_data["hovertext"], "x": fig_data["x"], "y": fig_data["y"]}
        )
        fig_data = fig_data.merge(data.loc[:, ["Name", "neighbor"]], on="Name")

        # plot points for neighbors
        neighbors_data = fig_data.loc[
            (fig_data["neighbor"]) & (fig_data["Name"] != advisor), :
        ]

        hovertemplate = (
            "<b>%{hovertext}</b><br><br>"
            + column
            + "=%{x}<br>Percentile=%{y}<extra></extra>"
        )

        fig.add_trace(
            go.Scattergl(
                x=neighbors_data["x"].values,
                y=neighbors_data["y"].values,
                hoverlabel=dict(bgcolor=red),
                hoverinfo="text",
                hovertext=neighbors_data["Name"].values,
                hovertemplate=hovertemplate,
                mode="markers",
                marker_color=red,
                marker_size=10,
                marker_line_color=dark_blue,
                marker_line_width=1,
                showlegend=False,
            )
        )

        # plot advisor as a star
        # plot points for neighbors
        advisor_data = fig_data.loc[fig_data["Name"] == advisor, :]

        fig.add_trace(
            go.Scattergl(
                x=advisor_data["x"].values,
                y=advisor_data["y"].values,
                hoverlabel=dict(bgcolor=dark_blue),
                hoverinfo="text",
                hovertext=advisor_data["Name"].values,
                hovertemplate=hovertemplate,
                mode="markers",
                marker_color=dark_blue,
                marker_size=18,
                marker_line_color="white",
                marker_line_width=1,
                marker_symbol="star",
                showlegend=False,
            )
        )

        fig.update_layout(
            xaxis_title=None,
            title=dict(
                text=f"<b>{column}</b>", x=0.5, xanchor="center", y=0.91, yanchor="top"
            ),
            yaxis=dict(title=dict(text="Percentile", font=dict(size=15))),
            font=dict(family="Roboto, sans-serif", color="#323232"),
        )

        if column == "Discretionary to Total Assets":
            fig.update_xaxes(tickformat=".1%")
        elif xaxis_type[column] == "linear":
            min_neighbor_y = fig_data.loc[fig_data["neighbor"], "y"].min()
            max_neighbor_y = fig_data.loc[fig_data["neighbor"], "y"].max()
            min_clip = fig_data.loc[fig_data["y"] > 1, "x"].min()
            max_clip = fig_data.loc[fig_data["y"] < 99, "x"].max()

            if max_neighbor_y >= 99:
                max_clip = fig_data["x"].max()
            if min_neighbor_y <= 1:
                min_clip = fig_data["x"].min()
            fig.update_xaxes(range=[min_clip, max_clip], tickformat=".1%")
        else:
            if column.find("Assets") != -1:
                fig.update_xaxes(tickprefix="$")
            fig.update_xaxes(type="log")
        return fig


con = db.connect(db_location)
cols = pd.read_sql("select * from reporting_data_formatted limit 1", con).columns
cols = cols.tolist()[1:]
con.close()

col_types = {
    "Name": "text",
    "State": "text",
    "Discretionary Assets": "numeric",
    "Non-Discretionary Assets": "numeric",
    "Total Assets": "numeric",
    "Discretionary Accounts": "numeric",
    "Non-Discretionary Accounts": "numeric",
    "Total Accounts": "numeric",
    "Financial Planning Clients": "text",
    "Advisory Employees": "numeric",
    "Investment Advisor Reps": "numeric",
    "Broker-Dealer Reps": "numeric",
    "Dually-Registered IARs": "numeric",
    "Insurance Agents": "numeric",
    "Outside Solicitors": "numeric",
    "Assets Per Advisor": "numeric",
    "Clients Per Advisor": "numeric",
    "Assets Per Client": "numeric",
    "Discretionary to Total Assets": "numeric",
    "Advisor Growth (1y)": "numeric",
    "Advisor Growth (3y)": "numeric",
    "Assets Growth (1y)": "numeric",
    "Assets Growth (3y)": "numeric",
    "Assets Per Advisor Growth (1y)": "numeric",
    "Assets Per Advisor Growth (3y)": "numeric",
    "Clients Per Advisor Growth (1y)": "numeric",
    "Clients Per Advisor Growth (3y)": "numeric",
    "Assets Per Client Growth (1y)": "numeric",
    "Assets Per Client Growth (3y)": "numeric",
    "Discretionary to Total Assets Growth (1y)": "numeric",
    "Discretionary to Total Assets Growth (3y)": "numeric",
}
col_formats = {
    "Discretionary Assets": FormatTemplate.money(0),
    "Non-Discretionary Assets": FormatTemplate.money(0),
    "Total Assets": FormatTemplate.money(0),
    "Discretionary Accounts": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Non-Discretionary Accounts": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Total Accounts": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Advisory Employees": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Investment Advisor Reps": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Broker-Dealer Reps": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Dually-Registered IARs": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Insurance Agents": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Outside Solicitors": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Assets Per Advisor": FormatTemplate.money(0),
    "Clients Per Advisor": Format(precision=0, group=",", scheme=Scheme.fixed),
    "Assets Per Client": FormatTemplate.money(0),
    "Discretionary to Total Assets": FormatTemplate.percentage(1),
    "Advisor Growth (1y)": FormatTemplate.percentage(1),
    "Advisor Growth (3y)": FormatTemplate.percentage(1),
    "Assets Growth (1y)": FormatTemplate.percentage(1),
    "Assets Growth (3y)": FormatTemplate.percentage(1),
    "Assets Per Advisor Growth (1y)": FormatTemplate.percentage(1),
    "Assets Per Advisor Growth (3y)": FormatTemplate.percentage(1),
    "Clients Per Advisor Growth (1y)": FormatTemplate.percentage(1),
    "Clients Per Advisor Growth (3y)": FormatTemplate.percentage(1),
    "Assets Per Client Growth (1y)": FormatTemplate.percentage(1),
    "Assets Per Client Growth (3y)": FormatTemplate.percentage(1),
    "Discretionary to Total Assets Growth (1y)": FormatTemplate.percentage(1),
    "Discretionary to Total Assets Growth (3y)": FormatTemplate.percentage(1),
}
col_selectable = {
    "Name": False,
    "State": False,
    "Discretionary Assets": True,
    "Non-Discretionary Assets": True,
    "Total Assets": True,
    "Discretionary Accounts": True,
    "Non-Discretionary Accounts": True,
    "Total Accounts": True,
    "Financial Planning Clients": False,
    "Advisory Employees": True,
    "Investment Advisor Reps": True,
    "Broker-Dealer Reps": True,
    "Dually-Registered IARs": True,
    "Insurance Agents": True,
    "Outside Solicitors": True,
    "Assets Per Advisor": True,
    "Clients Per Advisor": True,
    "Assets Per Client": True,
    "Discretionary to Total Assets": True,
    "Advisor Growth (1y)": True,
    "Advisor Growth (3y)": True,
    "Assets Growth (1y)": True,
    "Assets Growth (3y)": True,
    "Assets Per Advisor Growth (1y)": True,
    "Assets Per Advisor Growth (3y)": True,
    "Clients Per Advisor Growth (1y)": True,
    "Clients Per Advisor Growth (3y)": True,
    "Assets Per Client Growth (1y)": True,
    "Assets Per Client Growth (3y)": True,
    "Discretionary to Total Assets Growth (1y)": True,
    "Discretionary to Total Assets Growth (3y)": True,
}
datatable_cols = list()
for i in cols:
    col_dict = dict()
    col_dict["id"] = i
    col_dict["name"] = i
    col_dict["type"] = col_types[i]
    col_dict["selectable"] = col_selectable[i]
    if i in col_formats.keys():
        col_dict["format"] = col_formats[i]
    datatable_cols.append(col_dict)
app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>RIA Similarity App</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


app.layout = html.Div(
    className="container",
    children=[
        html.H1(
            children=[
                "RIA Similarity App",
                html.Span(
                    children=[
                        " by ",
                        html.A(href="http://jeremy-doyle.com", children="Jeremy Doyle"),
                    ]
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            className="row",
            children=[
                html.Div(
                    id="explainer",
                    className="six columns",
                    children=[
                        html.P(
                            children=[
                                "This application uses ",
                                html.A(
                                    "Form ADV Data",
                                    href="https://www.sec.gov/foia/docs/form-adv-archive-data.htm",
                                ),
                                """
                                obtained from the
                                US Securities and Exchange Commission (SEC) and is current
                                as of September 30, 2021.
                                """,
                            ]
                        ),
                        html.P(
                            """
                            Users may select a region and a Registered Investment
                            Advisor (RIA) and find up to 20 of the most similar 
                            RIAs within the selected region based on the data points 
                            presented below.
                            """
                        ),
                    ],
                ),
                html.Div(
                    className="six columns",
                    children=[
                        dcc.Dropdown(
                            id="region-select",
                            options=[
                                {"label": "All Regions", "value": "full"},
                                {"label": "New England", "value": "NewEngland"},
                                {"label": "Mid East", "value": "Mideast"},
                                {"label": "Southeast", "value": "Southeast"},
                                {"label": "Great Lakes", "value": "GreatLakes"},
                                {"label": "Plains", "value": "Plains"},
                                {"label": "Rocky Mountains", "value": "RockyMountain"},
                                {"label": "Southwest", "value": "Southwest"},
                                {"label": "Far West", "value": "FarWest"},
                                {"label": "Foreign", "value": "Foreign"},
                            ],
                            value="full",
                            clearable=False,
                        ),
                        dcc.Dropdown(
                            id="advisor-select",
                            placeholder="Select or Enter RIA Name",
                            value="",
                            options=[],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="twelve columns",
                    children=[
                        dcc.Slider(
                            id="slider",
                            min=1,
                            max=20,
                            marks={i: str(i) for i in range(1, 21)},
                            value=5,
                        )
                    ],
                )
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="twelve columns",
                    children=[
                        dash_table.DataTable(
                            id="table",
                            sort_action="none",
                            editable=False,
                            fixed_columns={"headers": True, "data": 1},
                            column_selectable="single",
                            selected_columns=["Discretionary Assets"],
                            style_table={"minWidth": "100%"},
                        )
                    ],
                )
            ],
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="twelve columns",
                    children=[dcc.Graph(id="ecdf-plot", figure=create_blank_fig())],
                )
            ],
        ),
    ],
)


@app.callback(Output("advisor-select", "options"), [Input("region-select", "value")])
def set_advisors(region):
    names = get_names(region)
    options = [{"label": name, "value": name} for name in names]
    return options


@app.callback(
    [Output("table", "data"), Output("table", "columns")],
    [
        Input("region-select", "value"),
        Input("advisor-select", "value"),
        Input("slider", "value"),
    ],
)
def update_rows(region, advisor, k):
    if advisor == "":
        data = {col["id"]: "" for col in datatable_cols}
        data["Name"] = "Please select or search for an RIA"
        data = [data]
        # data = [{'column':'Please select or search for an RIA'}]
    else:
        df = main(region, advisor, k)
        data = df.to_dict("records")
    columns = datatable_cols

    return data, columns


@app.callback(
    Output("ecdf-plot", "figure"),
    [
        Input("region-select", "value"),
        Input("advisor-select", "value"),
        Input("slider", "value"),
        Input("table", "selected_columns"),
    ],
)
def update_ecdf(region, advisor, k, column):
    column = column[0]
    fig = create_ecdf(region, advisor, k, column)
    return fig


server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
