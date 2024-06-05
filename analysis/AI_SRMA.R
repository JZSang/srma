library(dplyr)
library(ggplot2)
library(tidyr)

'#DDDBF1'
'#383F51'
'#BADEFC'
'#525252' 
'#B5446E'
'#9D44B5'
'#F55D3E'
'#76BED0'
'#5F0F40'
'#0F4C5C'
'#F87575'
"#E64B35"
"#4DBBD5"
"#00A087"
"#3C5488"
"#F39B7F"
"#8491B4"
"#91D1C2"
"#DC0000"
"#7E6148"
"#B09C85"

'#525252' 
'#BAB6E2'
'#D6FCFF'
'#BADEFC'
'#FA9E9E'
'#E95C49'
"#8491B4"
'#384F80'
'#50A2A7'
'#D44965'

abstract <- read.csv(file = "Figure_Datasets/Table1-AbstractPE.csv")
abstract_models <- read.csv(file = "Figure_Datasets/Fig2_models.csv")
ft_models <- read.csv(file = "Figure_Datasets/Fig3_models.csv")
abstract_val <- read.csv(file = "Figure_Datasets/Table2-AbstractGen.csv")
ft <- read.csv(file = "Figure_Datasets/Table3-FT_PE.csv")
ft_val <- read.csv(file = "Figure_Datasets/Table4-FT_Gen.csv")
human <- read.csv(file = "Figure_Datasets/Table0_human_vs_GPT.csv")
human_gpt <- read.csv(file = "Figure_Datasets/Table5-human_gpt_h2h.csv")

############################################################
# FIG 2 ABS PROMPT ENGINEERING

abstract$Model <- "GPT 4 (0125)"
Fig2_a <- unique(rbind(abstract, abstract_models))

Fig2_a <- Fig2_a %>%
  mutate(Model_Group = ifelse(Model == "GPT 4 (0125)", "GPT 4 (0125)", "Other Models"),
         Prompt_Model = paste(Prompt, Model),
         Prompt_Model = if_else(grepl("val", Dataset, ignore.case = TRUE), 
                                paste(Prompt_Model, "(Validation)"), 
                                Prompt_Model),
         SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN))*100,  # Calculating Standard Error for Sensitivity
         SE_Accuracy = sqrt(Accuracy * (1 - Accuracy) / (FP+TP+TN+FN))*100)

Fig2_a$Accuracy = Fig2_a$Accuracy * 100
Fig2_a_long_dotplot <- Fig2_a %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  arrange(Model_Group, Value) %>%
  mutate(Prompt_Model = factor(Prompt_Model, levels = unique(Prompt_Model)),
         Measure = factor(Measure, levels = c("Sensitivity", "Accuracy")))

Fig2_a_long_dotplot <- Fig2_a_long_dotplot %>%
  # Adding buffer rows just before the transition from GPT 4 (0125) to Other Models
  mutate(Prompt_Model = as.character(Prompt_Model)) %>%
  add_row(Prompt_Model = "Buffer 1", Model_Group = "Buffer", Measure = "Sensitivity", Value = NA) %>%
  add_row(Prompt_Model = "Buffer 1", Model_Group = "Buffer", Measure = "Accuracy", Value = NA) %>%
  mutate(Prompt_Model = factor(Prompt_Model, levels = c(levels(Fig2_a_long_dotplot$Prompt_Model), "Buffer 1")))

# Identify where to place buffer points exactly
transition_index <- 8
Fig2_a_long_dotplot <- Fig2_a_long_dotplot %>%
  arrange(Prompt_Model) %>%
  mutate(Prompt_Model = forcats::fct_inorder(Prompt_Model))
levels(Fig2_a_long_dotplot$Prompt_Model)
# Adjust factor levels to place the buffers correctly between GPT 4 (0125) and Other Models
levels_with_buffers <- c(levels(Fig2_a_long_dotplot$Prompt_Model)[1:3], 
                         levels(Fig2_a_long_dotplot$Prompt_Model)[5:7],
                         levels(Fig2_a_long_dotplot$Prompt_Model)[4],
                         levels(Fig2_a_long_dotplot$Prompt_Model)[8],
                         "Buffer 1", levels(Fig2_a_long_dotplot$Prompt_Model)[(transition_index + 1):length(levels(Fig2_a_long_dotplot$Prompt_Model))])
levels_with_buffers <- levels_with_buffers[1:15] # Change this as more models/evals are added
Fig2_a_long_dotplot$Prompt_Model <- factor(Fig2_a_long_dotplot$Prompt_Model, levels = levels_with_buffers)
levels_with_buffers

Fig2_a_long_dotplot <- Fig2_a_long_dotplot %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy))


pdf("fig_2/fig2a_lineplot.pdf", width = 13, height = 7)
ggplot(Fig2_a_long_dotplot, aes(x = Prompt_Model, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid") +  # Line for each measure
  geom_point(size = 3) +  # Points for each measure value
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96 * SE),
                width=0.1) + 
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  geom_vline(xintercept = 9, linetype = "dotted", color = "black", size = 1) +
  labs(x = "", y = "Performance (%)", title = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),  # Style x-axis text
    axis.text.y = element_text(face = "bold"),  # Style y-axis text
    axis.line = element_line(color = "black"),  # Add black lines for x and y axes
    panel.grid.major = element_blank(),  # Remove major gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    plot.title = element_text(hjust = 0.5),  # Center the title
    legend.position = c(0.932, 0.2),  # Adjust this to move the legend inside the plot
    legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
    legend.box.background = element_rect(color = "black", fill = "white"),
    text = element_text(size = 20),
    plot.margin = margin(t = 5, r = 10, b = 10, l = 0, unit = "pt")  # Adjust plot margins
  ) +
  scale_y_continuous(limits = c(29, 100), breaks = seq(30, 100, by = 10))+
  scale_x_discrete(labels = function(x) ifelse(x %in% c("Buffer 1", "Buffer 2"), "", x)) # Hide Buffer labels)  # Adjust left margin
dev.off()







############################################################
# FIG 2B ABS GENERALIZABILITY
library(binom)
abstract_val <- read.csv(file = "Figure_Datasets/Table2-AbstractGen.csv")
abstract_val$Accuracy <- abstract_val$Accuracy * 100

abstract_val <- abstract_val %>%
  mutate(
    Total_Sens = TP + FN,
    Total_Acc = TP + FP + TN + FN,
    CI_Sensitivity = binom.confint(x = TP, n = Total_Sens, conf.level = 0.95, methods = "exact"),
    CI_Accuracy = binom.confint(x = TP + TN, n = Total_Acc, conf.level = 0.95, methods = "exact"))
abstract_val <- abstract_val %>%
  mutate(
    Lower_CI_Sens = CI_Sensitivity$lower * 100,
    Upper_CI_Sens = CI_Sensitivity$upper * 100,
    Lower_CI_Acc = CI_Accuracy$lower * 100,
    Upper_CI_Acc = CI_Accuracy$upper * 100)

plot_data <- abstract_val %>%
  select(Dataset, Prompt, Sensitivity, Accuracy, Lower_CI_Sens, Upper_CI_Sens, Lower_CI_Acc, Upper_CI_Acc) %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Metric", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = c("Zero-shot", "ScreenPrompt")))  
plot_data <- plot_data %>%
  mutate(SE_lower = if_else(Metric == "Sensitivity", Lower_CI_Sens, Lower_CI_Acc),
         SE_upper = if_else(Metric == "Sensitivity", Upper_CI_Sens, Upper_CI_Acc))
# Create interaction term
plot_data <- plot_data %>%
  mutate(Dataset_Prompt = interaction(Dataset, Prompt, sep = " - ")) %>%
  arrange(Dataset, Prompt)  
dataset_prompts <- unique(plot_data$Dataset_Prompt)  
plot_data$Dataset_Prompt <- factor(plot_data$Dataset_Prompt, levels = dataset_prompts)  

#plotting a barplot
pdf("fig_2/fig2b_barplot.pdf", width = 10, height = 5)
dodge <- position_dodge(width = 0.8)
ggplot(plot_data, aes(x = Metric, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = SE_lower, ymax = SE_upper),
                position = dodge, width=0.25, color = "#707070") + 
  scale_fill_manual(values = c("#D6FCFF", "#50A2A7")) +
  labs(x = "", y = "Percentage (%)", title = "") +
  facet_wrap(~Dataset,nrow = 1) +
  theme_minimal() +
  theme(axis.line = element_line(color = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        strip.text.x = element_text(face = "bold"),
        legend.title = element_blank(),
        legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(0, 100, by = 20)) +
  coord_cartesian(ylim = c(0, 100))
dev.off()


############################################################
# FIG 3 FULL-TEXT PE
ft <- read.csv(file = "Figure_Datasets/Table3-FT_PE.csv")
ft_models <- read.csv(file = "Figure_Datasets/Fig3_models.csv")

ft$Model <- "GPT 4 (0125)"
Fig3_a <- unique(rbind(ft, ft_models))

Fig3_a <- Fig3_a %>%
  mutate(Model_Group = ifelse(Model == "GPT 4 (0125)", "GPT 4 (0125)", "Other Models"),
         Prompt_Model = paste(Prompt, Model),
         Prompt_Model = if_else(grepl("val", Dataset, ignore.case = TRUE), 
                                paste(Prompt_Model, "(Validation)"), 
                                Prompt_Model),
         SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN))*100,  # Calculating Standard Error for Sensitivity
         SE_Accuracy = sqrt(Accuracy * (1 - Accuracy) / (FP+TP+TN+FN))*100)

Fig3_a$Accuracy = Fig3_a$Accuracy * 100
Fig3_a_long_dotplot <- Fig3_a %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  arrange(Model_Group, Value) %>%
  mutate(Prompt_Model = factor(Prompt_Model, levels = unique(Prompt_Model)),
         Measure = factor(Measure, levels = c("Sensitivity", "Accuracy")))

Fig3_a_long_dotplot <- Fig3_a_long_dotplot %>%
  # Adding buffer rows just before the transition from GPT 4 (0125) to Other Models
  mutate(Prompt_Model = as.character(Prompt_Model)) %>%
  add_row(Prompt_Model = "Buffer 1", Model_Group = "Buffer", Measure = "Sensitivity", Value = NA) %>%
  add_row(Prompt_Model = "Buffer 1", Model_Group = "Buffer", Measure = "Accuracy", Value = NA) %>%
  mutate(Prompt_Model = factor(Prompt_Model, levels = c(levels(Fig3_a_long_dotplot$Prompt_Model), "Buffer 1")))

# Identify where to place buffer points exactly
transition_index <- 7
Fig3_a_long_dotplot <- Fig3_a_long_dotplot %>%
  arrange(Prompt_Model) %>%
  mutate(Prompt_Model = forcats::fct_inorder(Prompt_Model))
levels(Fig3_a_long_dotplot$Prompt_Model)
View(Fig3_a_long_dotplot)
# Adjust factor levels to place the buffers correctly between GPT 4 (0125) and Other Models
levels_with_buffers <- c(levels(Fig3_a_long_dotplot$Prompt_Model)[1:3], 
                         levels(Fig3_a_long_dotplot$Prompt_Model)[6],
                         levels(Fig3_a_long_dotplot$Prompt_Model)[4],
                         levels(Fig3_a_long_dotplot$Prompt_Model)[5],
                         levels(Fig3_a_long_dotplot$Prompt_Model)[7],
                         "Buffer 1", levels(Fig3_a_long_dotplot$Prompt_Model)[(transition_index + 1):length(levels(Fig3_a_long_dotplot$Prompt_Model))])
levels_with_buffers <- levels_with_buffers[1:13] # Change this as more models/evals are added
Fig3_a_long_dotplot$Prompt_Model <- factor(Fig3_a_long_dotplot$Prompt_Model, levels = levels_with_buffers)

Fig3_a_long_dotplot <- Fig3_a_long_dotplot %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy))

View(Fig3_a_long_dotplot)
pdf("fig_3/fig3a_lineplot.pdf", width = 12, height = 7)
ggplot(Fig3_a_long_dotplot, aes(x = Prompt_Model, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid") +  # Line for each measure
  geom_point(size = 3) +  # Points for each measure value
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96 * SE),
                width=0.1) + 
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  geom_vline(xintercept = 8, linetype = "dotted", color = "black", size = 1) +
  labs(x = "", y = "Performance (%)", title = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_blank(),  # Style x-axis text
    axis.text.y = element_text(face = "bold"),  # Style y-axis text
    axis.line = element_line(color = "black"),  # Add black lines for x and y axes
    panel.grid.major = element_blank(),  # Remove major gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    plot.title = element_text(hjust = 0.5),  # Center the title
    legend.position = c(0.932, 0.2),  # Adjust this to move the legend inside the plot
    legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
    legend.box.background = element_rect(color = "black", fill = "white"),
    text = element_text(size = 20),
    plot.margin = margin(t = 5, r = 10, b = 10, l = 0, unit = "pt")  # Adjust plot margins
  ) +
  scale_y_continuous(limits = c(30, 100), breaks = seq(30, 100, by = 10))+
  scale_x_discrete(labels = function(x) ifelse(x %in% c("Buffer 1", "Buffer 2"), "", x)) # Hide Buffer labels)  # Adjust left margin
dev.off()




############################################################
# FIG 3 FULL-TEXT GENERALIZABILITY
library(binom)
ft_val <- read.csv(file = "Figure_Datasets/Table4-FT_Gen.csv")
ft_val$Accuracy <- as.numeric(ft_val$Accuracy)
ft_val$Sensitivity <- as.numeric(ft_val$Sensitivity)
ft_val$Accuracy <- ft_val$Accuracy * 100

ft_val <- ft_val %>%
  mutate(
    Total_Sens = TP + FN,
    Total_Acc = TP + FP + TN + FN,
    CI_Sensitivity = binom.confint(x = TP, n = Total_Sens, conf.level = 0.95, methods = "exact"),
    CI_Accuracy = binom.confint(x = TP + TN, n = Total_Acc, conf.level = 0.95, methods = "exact"))
ft_val <- ft_val %>%
  mutate(
    Lower_CI_Sens = CI_Sensitivity$lower * 100,
    Upper_CI_Sens = CI_Sensitivity$upper * 100,
    Lower_CI_Acc = CI_Accuracy$lower * 100,
    Upper_CI_Acc = CI_Accuracy$upper * 100)

plot_data <- ft_val %>%
  select(Dataset, Prompt, Sensitivity, Accuracy, Lower_CI_Sens, Upper_CI_Sens, Lower_CI_Acc, Upper_CI_Acc) %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Metric", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = c("Zero-shot","ISO-ScreenPrompt")))  
plot_data <- plot_data %>%
  mutate(SE_lower = if_else(Metric == "Sensitivity", Lower_CI_Sens, Lower_CI_Acc),
         SE_upper = if_else(Metric == "Sensitivity", Upper_CI_Sens, Upper_CI_Acc))
plot_data <- plot_data %>%
  mutate(Dataset_Prompt = interaction(Dataset, Prompt, sep = " - ")) %>%
  arrange(Dataset, Prompt)  
dataset_prompts <- unique(plot_data$Dataset_Prompt)  
plot_data$Dataset_Prompt <- factor(plot_data$Dataset_Prompt, levels = dataset_prompts)  


pdf("fig_3/fig3b_barplot.pdf", width = 10, height = 4.8)
dodge <- position_dodge(width = 0.8)
ggplot(plot_data, aes(x = Metric, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = SE_lower, ymax = SE_upper),
                position=dodge,
                width=0.25,
                color = "#707070") + 
  scale_fill_manual(values = c("#D6FCFF", "#3C5488")) +
  labs(x = "", y = "Percentage (%)", title = "") +
  facet_wrap(~Dataset,nrow = 1) +
  theme_minimal() +
  theme(axis.line = element_line(color = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        strip.text.x = element_text(face = "bold"),
        legend.title = element_blank(),
        legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(0, 100, by = 20)) +
  coord_cartesian(ylim = c(0, 101))
dev.off()


############################################################
######## FIG 4 - HUMAN VS GPT

library(dplyr)
library(ggplot2)
library(dplyr)
library(ggplot2)
library(tidyr)


dual_ab <- read.csv(file = 'Figure_Datasets/Fig4-DualHumanAb.csv')

plot_data <- dual_ab %>%
  group_by(Dataset) %>%
  summarise(
    Human_Sensitivity = Sensitivity[Prompt == "Human"],
    ScreenPrompt_Sensitivity = Sensitivity[Prompt == "ScreenPrompt"],
    Human_Accuracy = Accuracy[Prompt == "Human"],
    ScreenPrompt_Accuracy = Accuracy[Prompt == "ScreenPrompt"],
    n_Human_Sens = TP[Prompt == "Human"] + FN[Prompt == "Human"],  
    n_ScreenPrompt_Sens = TP[Prompt == "ScreenPrompt"] + FN[Prompt == "ScreenPrompt"],  
    n_Human = TP[Prompt == "Human"] + TN[Prompt == "Human"] + FP[Prompt == "Human"] + FN[Prompt == "Human"],
    n_ScreenPrompt = TP[Prompt == "ScreenPrompt"] + TN[Prompt == "ScreenPrompt"] + FP[Prompt == "ScreenPrompt"] + FN[Prompt == "ScreenPrompt"]) %>%
  mutate(
    Sensitivity_Change = (ScreenPrompt_Sensitivity - Human_Sensitivity) * 100,
    Accuracy_Change = (ScreenPrompt_Accuracy - Human_Accuracy) * 100,
    SE_Human_Sensitivity = sqrt(Human_Sensitivity * (1 - Human_Sensitivity) / n_Human_Sens),
    SE_ScreenPrompt_Sensitivity = sqrt(ScreenPrompt_Sensitivity * (1 - ScreenPrompt_Sensitivity) / n_ScreenPrompt_Sens),
    SE_Sensitivity = sqrt(SE_Human_Sensitivity^2 + SE_ScreenPrompt_Sensitivity^2) * 100, # Calculate SE of the difference
    SE_Accuracy = sqrt((Human_Accuracy * (1 - Human_Accuracy) / n_Human) +
                         (ScreenPrompt_Accuracy * (1 - ScreenPrompt_Accuracy) / n_ScreenPrompt)) * 100) %>%
  pivot_longer(cols = c("Sensitivity_Change", "Accuracy_Change"),
               names_to = "Metric", 
               values_to = "Change") %>%
  mutate(
    Metric = factor(Metric, levels = c("Accuracy_Change", "Sensitivity_Change")),
    SE_Change = ifelse(grepl("Sensitivity", Metric), SE_Sensitivity, SE_Accuracy))


pdf("fig_4/fig4_forest_dual_abs.pdf", width = 7, height = 7)
ggplot(plot_data, aes(x = Change, y = Dataset, fill = Metric)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(aes(xmin = Change - SE_Change * 1.96, xmax = Change + SE_Change * 1.96),
                position = position_dodge(width = 0.7),
                color = "#707070", # Lighter gray color
                width = 0.25) +  # Adjusted for 95% CI
  scale_fill_manual(values = c("Sensitivity_Change" = "#525252", "Accuracy_Change" = "#BAB6E2")) +
  labs(x = "", y = "",
       title = "") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 12, face = "bold"),
        strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        legend.position = "none",
        panel.spacing = unit(2, "lines"))
dev.off()


##full text
h2h_ft <- read.csv(file = "Figure_Datasets/Table5-human_gpt_h2h.csv")

plot_data <- h2h_ft %>%
  group_by(Dataset) %>%
  summarise(
    Human_Sensitivity = Sensitivity[Prompt == "Human"],
    ScreenPrompt_Sensitivity = Sensitivity[Prompt == "ISO-ScreenPrompt"],
    Human_Accuracy = Accuracy[Prompt == "Human"],
    ScreenPrompt_Accuracy = Accuracy[Prompt == "ISO-ScreenPrompt"],
    n_Human_Sens = TP[Prompt == "Human"] + FN[Prompt == "Human"],  
    n_ScreenPrompt_Sens = TP[Prompt == "ISO-ScreenPrompt"] + FN[Prompt == "ISO-ScreenPrompt"],  # Same correction here
    n_Human = TP[Prompt == "Human"] + TN[Prompt == "Human"] + FP[Prompt == "Human"] + FN[Prompt == "Human"],
    n_ScreenPrompt = TP[Prompt == "ISO-ScreenPrompt"] + TN[Prompt == "ISO-ScreenPrompt"] + FP[Prompt == "ISO-ScreenPrompt"] + FN[Prompt == "ISO-ScreenPrompt"]) %>%
  mutate(
    Sensitivity_Change = (ScreenPrompt_Sensitivity - Human_Sensitivity) * 100,
    Accuracy_Change = (ScreenPrompt_Accuracy - Human_Accuracy) * 100,
    SE_Human_Sensitivity = sqrt(Human_Sensitivity * (1 - Human_Sensitivity) / n_Human_Sens),
    SE_ScreenPrompt_Sensitivity = sqrt(ScreenPrompt_Sensitivity * (1 - ScreenPrompt_Sensitivity) / n_ScreenPrompt_Sens),
    SE_Sensitivity = sqrt(SE_Human_Sensitivity^2 + SE_ScreenPrompt_Sensitivity^2) * 100, # Calculate SE of the difference
    SE_Accuracy = sqrt((Human_Accuracy * (1 - Human_Accuracy) / n_Human) +
                         (ScreenPrompt_Accuracy * (1 - ScreenPrompt_Accuracy) / n_ScreenPrompt)) * 100) %>%
  pivot_longer(cols = c("Sensitivity_Change", "Accuracy_Change"),
               names_to = "Metric", 
               values_to = "Change") %>%
  mutate(
    Metric = factor(Metric, levels = c("Accuracy_Change", "Sensitivity_Change")),
    SE_Change = ifelse(grepl("Sensitivity", Metric), SE_Sensitivity, SE_Accuracy))
View(plot_data)

pdf("fig_4/fig4_forest_ft.pdf", width = 7, height = 7)
ggplot(plot_data, aes(x = Change, y = Dataset, fill = Metric)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_col(position = "dodge", width = 0.7) +
  geom_errorbar(aes(xmin = Change - 1.96 * SE_Change, xmax = Change + 1.96 * SE_Change),
                position = position_dodge(width = 0.7),
                color = "#707070", # Lighter gray color
                width = 0.25) +  # Narrower spread of the caps) +  # Apply CI transformation here
  scale_fill_manual(values = c("Sensitivity_Change" = "#525252", "Accuracy_Change" = "#BAB6E2")) +
  labs(x = "", y = "",
       title = "") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size =12, face = "bold"),
        axis.text.x = element_text(size =12, face = "bold"),
        strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        legend.title = element_blank(),
        legend.position = "none",
        panel.spacing = unit(2, "lines")) +
  scale_x_continuous(
    breaks = seq(-20, max(plot_data$Change + 1.96 * plot_data$SE_Change, na.rm = TRUE), by = 20),
    limits = c(-20, max(plot_data$Change + 1.96 * plot_data$SE_Change, na.rm = TRUE))) 
dev.off()





############################################################
######## FIG 1 - HUMAN VS GPT (ST NLP)
human <- read.csv(file = "Figure_Datasets/Table0_human_vs_GPT.csv")
human <- human %>%
  mutate(
    SE_Sensitivity = sqrt(Sensitivity/100 * (1 - Sensitivity/100) / (TP + FN)),
    SE_Specificity = sqrt(Specificity/100 * (1 - Specificity/100) / (TN + FP)),
    SE_Accuracy = sqrt(Accuracy * (1 - Accuracy) / (TP + TN + FP + FN)),
    SE_Balanced_Accuracy = sqrt((SE_Sensitivity^2 + SE_Specificity^2) / 4)
  )
human$SE_Sensitivity = human$SE_Sensitivity * 100
human$SE_Specificity = human$SE_Specificity * 100
human$SE_Accuracy = human$SE_Accuracy * 100
human$SE_Balanced_Accuracy = human$SE_Balanced_Accuracy * 100

human_ft <- subset(human, Dataset == 'Full-text')
human_ab <- subset(human, Dataset == 'Abstract')

human_ft$Accuracy = human_ft$Accuracy * 100
human_ft_long_dotplot <- human_ft %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") 
human_ab$Accuracy = human_ab$Accuracy * 100
human_ab_long_dotplot <- human_ab %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") 

ft_prompt_order <- c("Probability Baseline", "Zero-shot", "Dual Human", "ISO-ScreenPrompt")
ab_prompt_order <- c("Probability Baseline", "Zero-shot", "Dual Human", "Abstract ScreenPrompt")

# Filtering to include only the specified prompts and then pivoting the data
human_ft_long_dotplot <- human_ft %>%
  filter(Prompt %in% ft_prompt_order) %>%
  pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy),
    Prompt = factor(Prompt, levels = ft_prompt_order))
human_ab_long_dotplot <- human_ab %>%
  filter(Prompt %in% ab_prompt_order) %>%
  pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy),
         Prompt = factor(Prompt, levels = ab_prompt_order))

### Full-text
# Plotting the data
pdf('fig_1/human_gpt_ft.pdf', width = 10, height = 7)
dodge_width <- 0
ggplot(human_ft_long_dotplot, aes(x = Prompt, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid", position = position_dodge(width = dodge_width)) +  # Apply dodging to lines
  geom_point(size = 3, position = position_dodge(width = dodge_width)) +  # Apply dodging to points
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96 * SE),
                width = 0.1, position = position_dodge(width = dodge_width)) +  # Ensure error bars are aligned
  geom_text(aes(label = sprintf("%.1f%%", Value)), position = position_nudge(y = -2), check_overlap = TRUE, hjust = -0.3, color = "black", size = 5) +
  # Displaying the prompt names on top of the data points for Accuracy with different nudging
  geom_text(data = filter(human_ft_long_dotplot, Measure == "Accuracy", Prompt == "Probability Baseline"),
            aes(label = Prompt), position = position_nudge(y = 11), check_overlap = TRUE, vjust = -0.5, color = "black", size = 5) +
  geom_text(data = filter(human_ft_long_dotplot, Measure == "Accuracy", Prompt == "Zero-shot"),
            aes(label = Prompt), position = position_nudge(y = 8), check_overlap = TRUE, vjust = -0.5, color = "black", size = 5) +
  geom_text(data = filter(human_ft_long_dotplot, Measure == "Accuracy", Prompt == "Dual Human"),
            aes(label = Prompt), position = position_nudge(y = 5), check_overlap = TRUE, vjust = -0.5, color = "black", size = 5) +
  geom_text(data = filter(human_ft_long_dotplot, Measure == "Accuracy", Prompt == "ISO-ScreenPrompt"),
            aes(label = Prompt), position = position_nudge(y = 5), check_overlap = TRUE, vjust = -0.5, color = "black", size = 5) +
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  labs(x = NULL, y = "Performance (%)", title = "Full-text Evaluation") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),  # Hide X-axis text
        axis.ticks.x = element_blank(),  # Hide X-axis ticks
        panel.grid.major = element_blank(),  # Hide major grid lines
        panel.grid.minor = element_blank(),  # Hide minor grid lines
        axis.line = element_line(color = "black"),  # Add black axis lines
        plot.title = element_text(hjust = 0.5),  # Center the title
        legend.position = c(0.85, 0.4),  # Adjust this to move the legend inside the plot
        legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
        legend.box.background = element_rect(color = "black", fill = "white"),
        text = element_text(size = 20)
  ) +
  scale_y_continuous(limits = c(10, 102), breaks = seq(20, 100, by = 20)) +
  theme(plot.margin = margin(t = 10, r = 5, b = 10, l = 10, unit = "pt"))
dev.off()


### Abstract
# Plotting the data
pdf('fig_1/human_gpt_ab.pdf', width = 10, height = 7)
ggplot(human_ab_long_dotplot, aes(x = Prompt, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid", position = position_dodge(width = dodge_width)) +  # Apply dodging to lines
  geom_point(size = 3, position = position_dodge(width = dodge_width)) +  # Apply dodging to points
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96 * SE),
                width = 0.1, position = position_dodge(width = dodge_width)) +  # Ensure error bars are aligned
  # Displaying the values next to each data point
  geom_text(
    data = filter(human_ab_long_dotplot, Measure == "Sensitivity"),
    aes(label = sprintf("%.1f%%", Value), y = ifelse(Prompt == "Abstract ScreenPrompt", Value + 2, Value - 2)),
    check_overlap = TRUE, hjust = -0.3, color = "black", size = 5
  ) +
  geom_text(
    data = filter(human_ab_long_dotplot, Measure == "Accuracy"),
    aes(label = sprintf("%.1f%%", Value), y = ifelse(Prompt == "Abstract ScreenPrompt", Value - 2, Value - 2)),
    check_overlap = TRUE, hjust = -0.3, color = "black", size = 5
  ) +
  # Displaying the prompt names on top of the data points for Accuracy with different nudging
  geom_text(data = filter(human_ab_long_dotplot, Measure == "Accuracy", Prompt == "Probability Baseline"),
            aes(label = Prompt), position = position_nudge(y = 11), check_overlap = TRUE, hjust = 0.5, color = "black", size = 5) +
  geom_text(data = filter(human_ab_long_dotplot, Measure == "Accuracy", Prompt == "Zero-shot"),
            aes(label = Prompt), position = position_nudge(y = 8), check_overlap = TRUE, hjust = 0.5, color = "black", size = 5) +
  geom_text(data = filter(human_ab_long_dotplot, Measure == "Accuracy", Prompt == "Dual Human"),
            aes(label = Prompt), position = position_nudge(y = 5), check_overlap = TRUE, hjust = 0.5, color = "black", size = 5) +
  geom_text(data = filter(human_ab_long_dotplot, Measure == "Accuracy", Prompt == "Abstract ScreenPrompt"),
            aes(label = Prompt), position = position_nudge(y = 9), check_overlap = TRUE, hjust = 0.5, color = "black", size = 5) +
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  labs(x = NULL, y = "Performance (%)", title = "Abstract Evaluation") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),  # Hide X-axis text
        axis.ticks.x = element_blank(),  # Hide X-axis ticks
        panel.grid.major = element_blank(),  # Hide major grid lines
        panel.grid.minor = element_blank(),  # Hide minor grid lines
        axis.line = element_line(color = "black"),  # Add black axis lines
        plot.title = element_text(hjust = 0.5),  # Center the title
        legend.position = c(0.85, 0.4),  # Adjust this to move the legend inside the plot
        legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
        legend.box.background = element_rect(color = "black", fill = "white"),
        text = element_text(size = 20)
  ) +
  scale_y_continuous(limits = c(10, 102), breaks = seq(20, 100, by = 20)) +
  theme(plot.margin = margin(t = 10, r = 5, b = 10, l = 10, unit = "pt"))
dev.off()
