rule all:
    input:
        "data/measles-evidence-vector-store.id",   
        "data/rugby_trace_hdi.nc",
        "data/rugby_trace.nc",
        "results/figure_2.txt",
        "figures/fig_2.png",
        "results/figure_3.txt",
        "figures/fig_4.png",
        "figures/fig_5.png",
        "figures/fig_6.png"

##################################
# Initialization

rule initialize_measles:
    input:
        "data/2405.09664v1.pdf" 
    output:
        "data/measles-evidence-vector-store.id" 
    shell:
        "pixi run python scripts/initialize.py --measles"

rule initialize_rugby:
    output:
        "data/rugby_trace_hdi.nc",
        "data/rugby_trace.nc"
    shell:
        "pixi run python scripts/initialize.py --rugby"

##################################
# Figure 2

rule run_figure_2_plot:
    input:
        "data/rugby_trace_hdi.nc",
        "data/rugby_trace.nc"
    output:
        "figures/fig_2.png"
    shell:
        "pixi run python scripts/figure_2.py --plot"   

rule run_figure_2_bumper:
    output:
        "results/figure_2.txt"
    shell:
        "pixi run python scripts/figure_2.py --bumper --iter 5"           

##################################
# Figure 3

rule run_figure_3_bumper:
    output:
        "results/figure_3.txt"
    shell:
        "pixi run python scripts/figure_3_bumper.py --iter 5"

##################################
# Figure 4

rule run_figure_4_experiments:
    output:
        "results/stability_base_timestamp.txt"
    shell:
        "pixi run python scripts/stability_experiments.py --experiment base"

rule run_figure_4_plot:
    input:
        "results/stability_base_timestamp.txt"
    output:
        "figures/fig_4.png"
    shell:
        "pixi run python scripts/figure_4.py"


##################################
# Figure 5

rule run_figure_5_experiments:
    output:
        "results/stability_element_timestamp.txt"
    shell:
        "pixi run python scripts/stability_experiments.py --experiment element"

rule run_figure_5_plot:
    input:
        "results/stability_element_timestamp.txt"
    output:
        "figures/fig_5.png"
    shell:
        "pixi run python scripts/figure_5.py"                

##################################
# Figure 6

rule run_figure_6_experiments:
    output:
        "results/state_space_timestamp.txt"
    shell:
        "pixi run python scripts/state_space_experiments.py"


rule run_figure_6_plot:
    input:
        "results/state_space_timestamp.txt"
    output:
        "figures/fig_6.png"
    shell:
        "pixi run python scripts/figure_6.py"      