\documentclass{article}
\usepackage{graphicx} % Required for inserting images

\title{Avelero_Carbon_Travel}
\author{Moussa Ouallaf}
\date{November 2025}

\usepackage{amsmath} 
\begin{document}

\maketitle

\section{Intorduction to the problem}
Currently, at our DDP hosting platform we request the user to provide minimal input values. We need to create a detailed carbon footprint based on these input values.
\\
\vspace{0.5cm}
The total carbon footprint can be formulated as:
\\
\vspace{0.2cm}
\[
\text{Total\_carbon\_footprint} = \text{material\_footprint} + \text{transport\_footprint}
\]

The material\_footprint can easily be defined as:
\[
\sum (\text{weight\_material} \times \text{carbon\_footprint\_material})
\]
\\
We ask the user for a detailed material list; therefore, these values can be calculated easily.
\\
The problem occurs during the calculation of the carbon footprint of transport. Many processes use different methods for transportation, and each method of transportation has their own carbon emission per km.
\\
This information will not be requested in our DDP hosting platform because many companies do not have access to such detailed data. Therefore, we need to generalize the method to allow a transport footprint calculation based on a rough indication of the total travel distance.

\section{Transportation methods}
To undermine this problem, we need to provide a clear overview of which transportation methods exist, what their emissions are, and when they are used:
\begin{table}[h]
    \centering
    \begin{tabular}{llr}
        \hline
        mode\_category & subtype & emission\_factor\_gCO2e\_per\_tkm \\
        \hline
        road  & Diesel articulated HGV $>33$ t, average laden     & 74    \\
        road  & Diesel articulated HGV $>3.5$--33 t, 100\% laden  & 67.63 \\
        road  & Road freight, generic, academic LCA               & 78    \\
        rail  & Freight train, generic                            & 22    \\
        inland\_waterway & Barge, generic                         & 31    \\
        sea   & Container ship, deep-sea, generic                 & 8.4   \\
        sea   & Container ship, short-sea, generic                & 16    \\
        air   & Long-haul air freight ($>3700$ km), freighter     & 560   \\
        air   & Long-haul air freight ($>3700$ km), belly freight & 990   \\
        \hline
    \end{tabular}
    \caption{Generic emission factors by transport mode and subtype used for default mode-level factors.}
\end{table}



\newpage
\vspace{5cm}
\section{generalised formula}

The base formula to calculate the carbon emission for transportation is:

\[
\underbrace{E(D)}_{\text{kg CO₂e}}
=
\left(
\underbrace{\frac{w}{1000}}_{\text{t}}
\right)
\cdot
\underbrace{D}_{\text{km}}
\cdot
\left(
\underbrace{
\frac{\sum_m s_m(D_m)\, EF_m}{1000}
}_{\text{kg CO₂e / tkm}}
\right)
\]


\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
\textbf{Symbol} & \textbf{Meaning} \\
\hline
$D$ & Travel Distance (km) \\
$w$ & Shipment weight (kg) \\
$m $ & Transport modes $m \in \{\text{road, rail, iww, sea, air}\}$ \\
$EF_m$ & Emission factor for mode $m$ (g CO$_2$e / tkm, WtW) \\
$s_m(D,\theta)$ & Share of tonne-km travelled by mode $m$, as a function of \\
& distance and context parameters $\theta$ \\
\hline
\end{tabular}
\end{table}

The challenging component is \(s_m(D,\theta)\). To evaluate this term, we must
identify which transport mode is used and the corresponding distance traveled
with this specific mode. However, this information is not collected on our DPP
hosting platform, as doing so would substantially increase the complexity of
data gathering. Therefore, we generalize this part of the equation on the basis
of validated findings in the empirical literature.


\newpage
\subsection{Generalisation of transortation methodes}
In the variables defined above, we already specified a generalized representation
of transportation modes. As shown in the transportation section, each mode
comprises several variants. For our purposes, we further abstract from these
variants and consider only the mode category
\\
\vspace{0.2cm}
\( m \in \{\text{road}, \text{rail}, \text{iww}, \text{sea}, \text{air}\} \).

\\
This can be done by calculating a fixed average for each mode category. 
By \emph{fixed average} we mean a weighted average based on the percentage 
usage of each subtype within that mode. For a given mode $m$ with subtypes 
$k \in K_m$, this gives us

\[
EF_m = \sum_{k \in K_m} EF_{m,k} \cdot u_{m,k},
\]

\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
\textbf{Symbol} & \textbf{Meaning} \\
\hline
$EF_{m,k}$   & Emission factor of subtype $k$ within mode $m$ \\
$u_{m,k}$    & Usage share of subtype $k$ within mode $m$, with $\sum_{k \in K_m} u_{m,k} = 1$ \\
$K_m$        & Set of subtypes belonging to mode $m$ \\
$EF_m$       & Weighted average emission factor for mode $m$ \\
\hline
\end{tabular}
\caption{Definition of quantities used in the weighted average emission factor calculation.}
\end{table}


By doing the calculations and research, we can dertermine the following values:
\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
\textbf{mode\_category} & \textbf{EF\_mode\_gCO$_2$e/tkm} \\
\hline
road              & 72.9 \\
rail              & 22.0 \\
inland\_waterway  & 31.0 \\
sea               & 10.3 \\
air               & 782.0 \\
\hline
\end{tabular}
\caption{Weighted-average emission factors per transport mode.}
\end{table}
\\
\vspace{0.2cm}
note: The calculations and sources of the weights of the usage share of the subtypes can be found in the csv file of the repository.

\subsection{Calculation of Weighted Emission Factor}

For the generalisation of the share of tonne-km travelled by different modes, 
we use a multinomial logit model. In this model, we calculate the probability 
of using a certain mode and normalise it by the attractiveness of all other 
modes. This formula can be expressed as:

\[
P_m(D) = \frac{\exp\!\big(U_m(D)\big)}{\displaystyle \sum_{k \in \text{all modes}} \exp\!\big(U_k(D)\big)},
\]


\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
\textbf{Symbol} & \textbf{Meaning} \\
\hline
$P_m(D)$ & Probability of using mode $m$ at distance $D$ \\
$U_m(D)$ & Utility (attractiveness) of mode $m$ at distance $D$ \\
$\exp(\cdot)$ & Exponential function \\
\hline
\end{tabular}
\end{table}

We obtain a mode-specific weight $P_m(D)$, which we use to multiply by the 
corresponding generalised emission factor. We do this for all transport modes
and sum the contributions:

\[
\overline{EF}(D) = \sum_{m \in \text{all modes}} P_m(D) \cdot EF_m.
\]
\\
\vspace{0.2cm}

To calculate the utility (attractiveness) of a mode, we specify $U_m(D)$ in 
a distance-dependent multinomial logit model. The advantage of this approach 
is that the attractiveness of a given mode depends on the attractiveness of 
the other modes. We use the following functional form:

\[
U_m(D) = \beta_{0,m} + \beta_{1,m} \ln D,
\]

where $D$ is the transport distance in kilometres and $\ln(\cdot)$ denotes the 
natural logarithm. Road is used as the reference mode, so its utility is 
normalised to zero: $U_{\text{road}}(D) \equiv 0$.

\begin{table}[h!]
\centering
\begin{tabular}{ll}
\hline
\textbf{Symbol} & \textbf{Meaning} \\
\hline
$D$             & Transport distance (km) \\
$U_m(D)$        & Utility (attractiveness) of mode $m$ at distance $D$ \\
$\beta_{0,m}$   & Mode-specific intercept (baseline attractiveness) \\
$\beta_{1,m}$   & Mode-specific sensitivity to distance (log-distance coefficient) \\
\hline
\end{tabular}
\end{table}

\\
\vspace{0.2 cm}
The values of $\beta_{1,m}$  and $\beta_{0,m}$ are research based by TU delft, however sea and air are reversed enginered. The thinking proces and sources to get this values can be found in the csv file. The current values are:





\end{document}