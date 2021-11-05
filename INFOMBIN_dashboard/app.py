"""
Created on Monday October 20 2021
@author: Angel Temelko
"""
from _plotly_utils.png import Image
import dash
from dash import dcc
from dash import html
from numpy.lib.utils import source
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from data_preparation_descriptive_part import trainData,holiday_sales_fig,average_sales_dep_fig,average_sales_store_fig,cpi_and_sales_fig,fuel_and_sales_fig,size_and_sales_fig,unemployment_and_sales_fig,week_of_year_and_sales_fig,weekly_sales_fig,size_and_temperature_fig,pieFigData
from data_prediction import weekly_sales_fig_pred,monthly_sales_fig_pred

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('__name__', external_stylesheets=external_stylesheets)

pieFig1 = px.pie(pieFigData, values='Counts', names='Store_Type',title='Popularity of Store Types',labels='Store_Type')

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H3(children="Walmart",
                        style={
                            "margin-bottom": "0px",
                            "color": "white"}
                        ),
                html.H6(children="Visualize sales forcasting for walmart.",
                        style={
                            "margin-top": "0px",
                            "color": "white"}
                        ),
            ])
        ], id="title"),
    ], id="header"),

    html.Div([
        html.Div([
            html.H6(children="Stores:",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(trainData['Store'].iloc[-1],
                   id="stores",
                   style={
                       "textAlign": "center",
                       "color": "white",
                       "fontSize": 30}
                   )], className="card_container three columns"
        ),
        html.Div([
            html.H6(children="Departments",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(trainData['Dept'].iloc[-1] + 1,
                   id="departments",
                   style={
                       "textAlign": "center",
                       "color": "white",
                       "fontSize": 30}
                   )], className="card_container three columns"
        ),
        html.Div([
            html.H6(children="Weekly sales - Mean",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(trainData['Weekly_Sales'].mean().round(1),
                   id="Weekly sales",
                   style={
                       "textAlign": "center",
                       "color": "white",
                       "fontSize": 30}
                   )
        ], className="card_container three columns"
        ),
        html.Div([
            html.H6(children="Weekly sales - Median",
                    style={
                        "textAlign": "center",
                        "color": "white"}
                    ),
            html.P(trainData['Weekly_Sales'].median().round(1),
                   id="pop",
                   style={
                       "textAlign": "center",
                       "color": "white",
                       "fontSize": 30}
                   )], className="card_container three columns"
        )
    ], className="row flex-display"),
    html.Div([
        html.Div([
            html.P("Forecasting sales:", className="fix_label", style={"color": "white"}),
            dcc.Dropdown(id="myDropDown",
                         multi=False,
                         clearable=False,
                         value="Monthly sales",
                         placeholder="Forecasting",
                         options=[{"label": c, "value": c}
                                  for c in ["Monthly sales", "Weekly sales"]],
                         className="dcc_compon"),
            html.Div([
                dcc.Graph(id="anotherLineChart"),
            ],),
        ], className="create_container six columns"),
        html.Div([
            html.P("Average sales:", className="fix_label", style={"color": "white"}),
            dcc.Dropdown(id="mode",
                         multi=False,
                         clearable=False,
                         value="Average Sales per Store",
                         placeholder="Descriptive graphs",
                         options=[{"label": c, "value": c}
                                  for c in ['Average Sales per Store', 'Average sales per Department', 'Sales based on holiday','Average weekly sales per year']],
                         className="dcc_compon"),
            html.Div([
            dcc.Graph(id="line_chart")
            ],),
        ], className="create_container six columns"),
    ], className="row flex-display"),
    html.Div([
        html.Div([
            html.P("Populairty of stores", className="fix_label", style={"color": "white"}),
            dcc.Graph(id="pieChart",figure=pieFig1,style={"margin-top": "45px"})
        ], className="create_container six columns"),
        html.Div([
            html.P("Coorleations beetween values:", className="fix_label", style={"color": "white"}),
            dcc.Dropdown(id="mode2",
                         multi=False,
                         clearable=False,
                         value="Correleation between week of year and sales",
                         placeholder="Coorleations beetween values",
                         options=[{"label": c, "value": c}
                                  for c in ['Correleation between week of year and sales', 'Correleation between Size and sales', 'Correleation between Temperature and sales',
                                   'Correleation between Fuel Price and sales', 'Correleation between CPI and sales', 'Correleation between Unemploymen and sales']],
                         className="dcc_compon"),
            html.Div([
            dcc.Graph(id="line_chart_coorelation")
            ],),
        ], className="create_container six columns"),
    ], className="row flex-display")
])

@app.callback(Output("anotherLineChart", "figure"),
             [Input("myDropDown", "value")])
def update_figure(myDropDown):
    if myDropDown == 'Monthly sales':
        return monthly_sales_fig_pred
    elif myDropDown == 'Weekly sales':
        return weekly_sales_fig_pred


@app.callback(Output("line_chart", "figure"),
             [Input("mode", "value")])
def update_figure(mode):
    if mode == 'Average Sales per Store':
        return average_sales_store_fig
    elif mode == 'Average sales per Department':
        return average_sales_dep_fig
    elif mode == 'Sales based on holiday':
        return holiday_sales_fig
    elif mode == 'Average weekly sales per year':
        return weekly_sales_fig

@app.callback(Output("line_chart_coorelation", "figure"),
             [Input("mode2", "value")])
def update_figure(mode2):
    if mode2 == 'Average weekly sales per year':
        return weekly_sales_fig
    elif mode2 == 'Correleation between week of year and sales':
        return week_of_year_and_sales_fig
    elif mode2 == 'Correleation between Size and sales':
        return size_and_sales_fig
    elif mode2 == 'Correleation between Temperature and sales':
        return size_and_temperature_fig
    elif mode2 == 'Correleation between Fuel Price and sales':
        return fuel_and_sales_fig
    elif mode2 == 'Correleation between CPI and sales':
        return cpi_and_sales_fig
    elif mode2 == 'Correleation between Unemploymen and sales':
        return unemployment_and_sales_fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
