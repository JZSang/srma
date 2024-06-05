library(dplyr)
library(ggplot2)
library(tidyr)

'#525252' 
'#BAB6E2'
'#D6FCFF'
'#BADEFC'
'#FA9E9E'
'#E95C49'
"#8491B4"
'#384F80'
'#263554'
'#50A2A7'
'#285053'
'#D44965'

abstract_pe <- read.csv(file = "Supp_Figure_Dataset/FigS1-AbstractPE.csv")
fewshot <- read.csv(file = "Supp_Figure_Dataset/FigS1-FewShotPE.csv")
ab_ROB <- read.csv(file = "Supp_Figure_Dataset/FigS1-RoB.csv")
ft_PE <- read.csv(file = "Supp_Figure_Dataset/FigS2-FTPE.csv")
ISO_AB <- read.csv(file = "Supp_Figure_Dataset/FigS2-ISOAb.csv")
ft_ROB <- read.csv(file = "Supp_Figure_Dataset/FigS2-RoB.csv")
sc_gen_ab <- read.csv(file = "Supp_Figure_Dataset/FigS1-sc.csv")
sc_gen_ft <- read.csv(file = "Supp_Figure_Dataset/FigS2-sc.csv")


############################################################
# FIG S1 Abstract PE
abstract_pe <- read.csv(file = "Supp_Figure_Dataset/FigS1-AbstractPE.csv")
abstract_pe$Accuracy <- abstract_pe$Accuracy*100
ab <- abstract_pe %>%
  mutate(SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,  # SE for Sensitivity
         SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (FP+TP+TN+FN)) * 100)  # SE for Accuracy
ab_long <- ab %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = unique(Prompt)),
         Measure = factor(Measure, levels = c("Sensitivity", "Accuracy")))
ab_long <- ab_long %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy))


pdf("supp_fig/figs1/abstract_pe.pdf", width = 5, height = 5)
ggplot(ab_long, aes(x = Prompt, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid") +  # Line for each measure
  geom_point(size = 3) +  # Points for each measure value
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96* SE),
                width=0.05) + 
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  geom_vline(xintercept = 8.5, linetype = "dotted", color = "black", size = 1) +
  labs(x = "", y = "Performance (%)", title = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),  # Style x-axis text
    axis.text.y = element_text(face = "bold"),  # Style y-axis text
    axis.line = element_line(color = "black"),  # Add black lines for x and y axes
    panel.grid.major = element_blank(),  # Remove major gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    plot.title = element_text(hjust = 0.5),  # Center the title
    legend.position = c(0.875, 0.28),  # Adjust this to move the legend inside the plot
    legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
    legend.box.background = element_rect(color = "black", fill = "white"),
    text = element_text(size = 13),
    plot.margin = margin(t = 0, r = 10, b = 10, l = 0, unit = "pt")  # Adjust plot margins
  ) +
  scale_y_continuous(limits = c(80, 100), breaks = seq(80, 100, by = 10))+
  scale_x_discrete(labels = function(x) ifelse(x %in% c("Buffer 1", "Buffer 2"), "", x)) # Hide Buffer labels)  # Adjust left margin
dev.off()

scale_y_continuous(expand = expansion(mult = 0), breaks = seq(40, 100, by = 20))

############################################################
# FIG S1 Few-shot PE
fewshot <- read.csv(file = "Supp_Figure_Dataset/FigS1-FewShotPE.csv")

fewshot$Accuracy <- fewshot$Accuracy*100
fs_pe <- fewshot %>%
  mutate(SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,  # SE for Sensitivity
         SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (FP+TP+TN+FN)) * 100)  # SE for Accuracy
fs_pe_long <- fs_pe %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = unique(Prompt)),
         Measure = factor(Measure, levels = c("Sensitivity", "Accuracy")))
fs_pe_long <- fs_pe_long %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy))

pdf("supp_fig/figs1/fewshot_pe.pdf", width = 5, height = 5)
ggplot(fs_pe_long, aes(x = Prompt, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid") +  # Line for each measure
  geom_point(size = 3) +  # Points for each measure value
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96* SE),
                width=0.05) + 
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  geom_vline(xintercept = 8.5, linetype = "dotted", color = "black", size = 1) +
  labs(x = "", y = "Performance (%)", title = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),  # Style x-axis text
    axis.text.y = element_text(face = "bold"),  # Style y-axis text
    axis.line = element_line(color = "black"),  # Add black lines for x and y axes
    panel.grid.major = element_blank(),  # Remove major gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    plot.title = element_text(hjust = 0.5),  # Center the title
    legend.position = c(0.875, 0.60),  # Adjust this to move the legend inside the plot
    legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
    legend.box.background = element_rect(color = "black", fill = "white"),
    text = element_text(size = 13),
    plot.margin = margin(t = 0, r = 10, b = 10, l = 0, unit = "pt")  # Adjust plot margins
  ) +
  scale_y_continuous(limits = c(80, 100.2), breaks = seq(80, 100, by = 10))+
  scale_x_discrete(labels = function(x) ifelse(x %in% c("Buffer 1", "Buffer 2"), "", x)) # Hide Buffer labels)  # Adjust left margin
dev.off()



############################################################
# FIG Self-consistency gain

sc_gen_ab <- read.csv(file = "Supp_Figure_Dataset/FigS1-sc.csv")
sc_gen_ab$Accuracy <- as.numeric(sc_gen_ab$Accuracy)
sc_gen_ab$Sensitivity <- as.numeric(sc_gen_ab$Sensitivity)
sc_gen_ab$Accuracy <- sc_gen_ab$Accuracy * 100

sc_gen_ab <- sc_gen_ab %>%
  mutate(
    SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,
    SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (TP+FP+TN+FN)) * 100)

plot_data <- sc_gen_ab %>%
  select(Dataset, Prompt, Sensitivity, Accuracy, SE_Sensitivity, SE_Accuracy) %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Metric", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = c("ScreenPrompt","ScreenPrompt-SC")))  
plot_data <- plot_data %>%
  mutate(SE = if_else(Metric == "Sensitivity", SE_Sensitivity, SE_Accuracy))
plot_data <- plot_data %>%
  mutate(Dataset_Prompt = interaction(Dataset, Prompt, sep = " - ")) %>%
  arrange(Dataset, Prompt)  
dataset_prompts <- unique(plot_data$Dataset_Prompt)  
plot_data$Dataset_Prompt <- factor(plot_data$Dataset_Prompt, levels = dataset_prompts)  


pdf("supp_fig/figs1/sc_gen.pdf", width = 6, height = 6)
dodge <- position_dodge(width = 0.8)
ggplot(plot_data, aes(x = Metric, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96* SE),
                position=dodge,
                width=0.25,
                color = "#707070") + 
  scale_fill_manual(values = c("#50A2A7", "#285053")) +
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
        legend.position = "none",
        plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
        text = element_text(size = 13)) +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(70, 100, by = 10)) +
  coord_cartesian(ylim = c(70, 101))
dev.off()

############################################################
# FIG S1 AUROC Curve
library(dplyr)
library(pROC)
library(stringr)
library(ggplot2)

serotracker <- "Supp_Figure_Dataset/ensemble_st_ab.csv"
reinfection <- "Supp_Figure_Dataset/ensemble_re_ab.csv"

df <- read.csv(serotracker)
df$included_count <- sapply(df$Mode, function(x) {
  # Extract all occurrences of 'included' and 'excluded'
  votes <- str_extract_all(x, "\\b(included|excluded)\\b")[[1]]
  # Count the number of times 'included' appears
  sum(votes == "included")
})

df$Actual.value <- ifelse(df$Actual.value == "included", 1, 0)

# Prepare results dataframe
results <- data.frame(Sensitivity = numeric(), Specificity = numeric(), Threshold = numeric())

# Calculate metrics for each threshold
for (threshold in 0:12) {
  predicted_binary <- as.numeric(df$included_count >= threshold)
  # ROC calculation
  roc_curve <- roc(response = df$Actual.value, predictor = predicted_binary)
  auc_value <- auc(roc_curve)
  # Sensitivity and Specificity calculation
  tp <- sum(predicted_binary == 1 & df$Actual.value == 1)
  fn <- sum(predicted_binary == 0 & df$Actual.value == 1)
  tn <- sum(predicted_binary == 0 & df$Actual.value == 0)
  fp <- sum(predicted_binary == 1 & df$Actual.value == 0)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  # Store results
  results <- rbind(results, data.frame(Sensitivity = sensitivity, Specificity = specificity, Threshold = threshold, AUC = auc_value))
}

# Plotting the ROC curve
pdf("supp_fig/figs1/AUROC_ST.pdf", width = 6, height = 4)
ggplot(results, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_line(color = "black", size = 1) +  # Smooth line with specified color
  labs(title = "", 
       x = "False Positive Rate (1 - Specificity)", 
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(
   axis.line = element_line(color = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        strip.text.x = element_text(face = "bold"),
        text = element_text(size = 13),
        legend.title = element_blank(),
        legend.position = "right")
dev.off()

roc_curve <- roc(df$Actual.value, df$included_count)
auc_value <- auc(roc_curve)
cat("AUC Value: ", auc_value, "\n") # Print the AUROC value





# FIG S2 Full-text PE
ft_PE <- read.csv(file = "Supp_Figure_Dataset/FigS2-FTPE.csv")
ft_PE$Accuracy <- ft_PE$Accuracy*100
ft <- ft_PE %>%
  mutate(SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,  # SE for Sensitivity
         SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (FP+TP+TN+FN)) * 100)  # SE for Accuracy
ft_long <- ft %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = unique(Prompt)),
         Measure = factor(Measure, levels = c("Sensitivity", "Accuracy")))
ft_long <- ft_long %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy))


pdf("supp_fig/figs2/ft_PE.pdf", width = 6, height = 6)
ggplot(ft_long, aes(x = Prompt, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid") +  # Line for each measure
  geom_point(size = 3) +  # Points for each measure value
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96* SE),
                width=0.05) + 
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  geom_vline(xintercept = 8.5, linetype = "dotted", color = "black", size = 1) +
  labs(x = "", y = "Performance (%)", title = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),  # Style x-axis text
    axis.text.y = element_text(face = "bold"),  # Style y-axis text
    axis.line = element_line(color = "black"),  # Add black lines for x and y axes
    panel.grid.major = element_blank(),  # Remove major gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    plot.title = element_text(hjust = 0.5),  # Center the title
    legend.position = c(0.88, 0.3),  # Adjust this to move the legend inside the plot
    legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
    legend.box.background = element_rect(color = "black", fill = "white"),
    text = element_text(size = 14),
    plot.margin = margin(t = 0, r = 10, b = 10, l = 0, unit = "pt")  # Adjust plot margins
  ) +
  scale_y_continuous(limits = c(78, 100), breaks = seq(80, 100, by = 10))+
  scale_x_discrete(labels = function(x) ifelse(x %in% c("Buffer 1", "Buffer 2"), "", x)) # Hide Buffer labels)  # Adjust left margin
dev.off()

scale_y_continuous(expand = expansion(mult = 0), breaks = seq(40, 100, by = 20))

############################################################
# FIG S2 ISO-Ab
ISO_AB <- read.csv(file = "Supp_Figure_Dataset/FigS2-ISOAb.csv")

ISO_AB$Accuracy <- ISO_AB$Accuracy*100
ISO_AB <- ISO_AB %>%
  mutate(SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,  # SE for Sensitivity
         SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (FP+TP+TN+FN)) * 100)  # SE for Accuracy
ISO_AB_long <- ISO_AB %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Measure", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = unique(Prompt)),
         Measure = factor(Measure, levels = c("Sensitivity", "Accuracy")))
ISO_AB_long <- ISO_AB_long %>%
  mutate(SE = if_else(Measure == "Sensitivity", SE_Sensitivity, SE_Accuracy))

pdf("supp_fig/figs2/ISO_ab.pdf", width = 6, height = 6)
ggplot(ISO_AB_long, aes(x = Prompt, y = Value, group = Measure, color = Measure)) +
  geom_line(size = 1, linetype = "solid") +  # Line for each measure
  geom_point(size = 3) +  # Points for each measure value
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96* SE),
                width=0.05) + 
  scale_color_manual(values = c("Accuracy" = "#BAB6E2", "Sensitivity" = "#525252")) +
  geom_vline(xintercept = 8.5, linetype = "dotted", color = "black", size = 1) +
  labs(x = "", y = "Performance (%)", title = "") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),  # Style x-axis text
    axis.text.y = element_text(face = "bold"),  # Style y-axis text
    axis.line = element_line(color = "black"),  # Add black lines for x and y axes
    panel.grid.major = element_blank(),  # Remove major gridlines
    panel.grid.minor = element_blank(),  # Remove minor gridlines
    plot.title = element_text(hjust = 0.5),  # Center the title
    legend.position = c(0.88, 0.3),  # Adjust this to move the legend inside the plot
    legend.background = element_rect(fill = "white", colour = "black", size = 0.5),  # Optional: Add background to legend
    legend.box.background = element_rect(color = "black", fill = "white"),
    text = element_text(size = 14),
    plot.margin = margin(t = 0, r = 10, b = 10, l = 0, unit = "pt")  # Adjust plot margins
  ) +
  scale_y_continuous(limits = c(80, 100), breaks = seq(80, 100, by = 10))+
  scale_x_discrete(labels = function(x) ifelse(x %in% c("Buffer 1", "Buffer 2"), "", x)) # Hide Buffer labels)  # Adjust left margin
dev.off()


############################################################
# FIGS2 Self-consistency gain

sc_gen_ft <- read.csv(file = "Supp_Figure_Dataset/FigS2-sc.csv")
sc_gen_ft$Accuracy <- as.numeric(sc_gen_ft$Accuracy)
sc_gen_ft$Sensitivity <- as.numeric(sc_gen_ft$Sensitivity)
sc_gen_ft$Accuracy <- sc_gen_ft$Accuracy * 100

sc_gen_ft <- sc_gen_ft %>%
  mutate(SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,
         SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (TP+FP+TN+FN)) * 100)

plot_data <- sc_gen_ft %>%
  select(Dataset, Prompt, Sensitivity, Accuracy, SE_Sensitivity, SE_Accuracy) %>%
  tidyr::pivot_longer(cols = c("Sensitivity", "Accuracy"), names_to = "Metric", values_to = "Value") %>%
  mutate(Prompt = factor(Prompt, levels = c("ISO-ScreenPrompt","ISO-ScreenPrompt-SC")))  
plot_data <- plot_data %>%
  mutate(SE = if_else(Metric == "Sensitivity", SE_Sensitivity, SE_Accuracy))
plot_data <- plot_data %>%
  mutate(Dataset_Prompt = interaction(Dataset, Prompt, sep = " - ")) %>%
  arrange(Dataset, Prompt)  
dataset_prompts <- unique(plot_data$Dataset_Prompt)  
plot_data$Dataset_Prompt <- factor(plot_data$Dataset_Prompt, levels = dataset_prompts)  


pdf("supp_fig/figs2/sc_gen.pdf", width = 6, height = 6)
dodge <- position_dodge(width = 0.8)
ggplot(plot_data, aes(x = Metric, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = Value + 1.96* SE),
                position=dodge,
                width=0.25,
                color = "#707070") + 
  scale_fill_manual(values = c("#3C5488", "#263554")) +
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
        legend.position = "nonee",
        plot.margin = margin(t = 0, r = 0, b = 0, l = 0, unit = "pt"),
        text = element_text(size = 14)) +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(70, 100, by = 10)) +
  coord_cartesian(ylim = c(70, 101))
dev.off()

############################################################
# FIG S2 AUROC Curve
library(dplyr)
library(pROC)
library(stringr)
library(ggplot2)

serotracker <- "Supp_Figure_Dataset/ensemble_st_ft.csv"
reinfection <- "Supp_Figure_Dataset/ensemble_re_ft.csv"

df <- read.csv(serotracker)
df$included_count <- sapply(df$Mode, function(x) {
  # Extract all occurrences of 'included' and 'excluded'
  votes <- str_extract_all(x, "\\b(included|excluded)\\b")[[1]]
  # Count the number of times 'included' appears
  sum(votes == "included")
})

df$Actual.value <- ifelse(df$Actual.value == "included", 1, 0)

# Prepare results dataframe
results <- data.frame(Sensitivity = numeric(), Specificity = numeric(), Threshold = numeric())

# Calculate metrics for each threshold
for (threshold in 0:12) {
  predicted_binary <- as.numeric(df$included_count >= threshold)
  # ROC calculation
  roc_curve <- roc(response = df$Actual.value, predictor = predicted_binary)
  auc_value <- auc(roc_curve)
  # Sensitivity and Specificity calculation
  tp <- sum(predicted_binary == 1 & df$Actual.value == 1)
  fn <- sum(predicted_binary == 0 & df$Actual.value == 1)
  tn <- sum(predicted_binary == 0 & df$Actual.value == 0)
  fp <- sum(predicted_binary == 1 & df$Actual.value == 0)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  # Store results
  results <- rbind(results, data.frame(Sensitivity = sensitivity, Specificity = specificity, Threshold = threshold, AUC = auc_value))
}

# Plotting the ROC curve
pdf("supp_fig/figs2/AUROC_ST.pdf", width = 6, height = 4)
ggplot(results, aes(x = 1 - Specificity, y = Sensitivity)) +
  geom_line(color = "black", size = 1) +  # Smooth line with specified color
  labs(title = "", 
       x = "False Positive Rate (1 - Specificity)", 
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal() +
  theme(
    axis.line = element_line(color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    strip.text.x = element_text(face = "bold"),
    text = element_text(size = 13),
    legend.title = element_blank(),
    legend.position = "right")
dev.off()

roc_curve <- roc(df$Actual.value, df$included_count)
auc_value <- auc(roc_curve)
cat("AUC Value: ", auc_value, "\n") # Print the AUROC value









############################################################
# FIG S3 Single Abstract 

human_gpt_ab <- read.csv(file = "Figure_Datasets/Table5-human_gpt_h2h_abs.csv")
human_gpt_ab$Accuracy <- as.numeric(human_gpt_ab$Accuracy) * 100
human_gpt_ab$Sensitivity <- as.numeric(human_gpt_ab$Sensitivity) * 100

human_gpt_ab <- human_gpt_ab %>%
  mutate(
    SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,
    SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (TP+FP+TN+FN)) * 100)

plot_data <- human_gpt_ab %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity"), names_to = "Metric", values_to = "Value")
plot_data$Metric <- factor(plot_data$Metric, levels = c("Sensitivity", "Accuracy"))
plot_data <- plot_data %>%
  mutate(SE = if_else(Metric == "Sensitivity", SE_Sensitivity, SE_Accuracy))

pdf("supp_fig/figs3/single_human_bar.pdf", width = 5, height = 7)
ggplot(plot_data, aes(x = Dataset, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.7) +
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = pmin(Value + 1.96 * SE, 100)),
                position=position_dodge(width = 0.7),
                width=0.25,
                color = "#707070") + 
  scale_fill_manual(values = c("Human" = "#EB6C5C", "ScreenPrompt" = "#50A2A7")) +
  labs(x = "Dataset", y = "Percentage (%)", title = "Single Human vs. Abstract ScreenPrompt") +
  facet_wrap(~ Metric, scales = "free", ncol = 1) +
  theme_minimal() +
  theme(axis.line = element_line(color = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        strip.text.x = element_text(face = "bold"),
        legend.title = element_blank(),
        legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(0, 100, by = 10)) +
  coord_cartesian(ylim = c(70, 101))  # Setting the visible area without dropping data
dev.off()


############################################################
# FIG S3 Dual Abstract 
dual_ab <- read.csv(file = 'Figure_Datasets/Fig4-DualHumanAb.csv')

dual_ab$Accuracy <- as.numeric(dual_ab$Accuracy) * 100
dual_ab$Sensitivity <- as.numeric(dual_ab$Sensitivity) * 100

dual_ab <- dual_ab %>%
  mutate(
    SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,
    SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (TP+FP+TN+FN)) * 100)

plot_data <- dual_ab %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity"), names_to = "Metric", values_to = "Value")
plot_data$Metric <- factor(plot_data$Metric, levels = c("Sensitivity", "Accuracy"))
plot_data <- plot_data %>%
  mutate(SE = if_else(Metric == "Sensitivity", SE_Sensitivity, SE_Accuracy))

pdf("supp_fig/figs3/dual_human_ab_bar.pdf", width = 5, height = 7)
ggplot(plot_data, aes(x = Dataset, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.7) +
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = pmin(Value + 1.96 * SE, 100)),
                position=position_dodge(width = 0.7),
                width=0.25,
                color = "#707070") + 
  scale_fill_manual(values = c("Human" = "#DB4345", "ScreenPrompt" = "#50A2A7")) +
  labs(x = "Dataset", y = "Percentage (%)", title = "Dual Human vs. Abstract ScreenPrompt") +
  facet_wrap(~ Metric, scales = "free", ncol = 1) +
  theme_minimal() +
  theme(axis.line = element_line(color = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        strip.text.x = element_text(face = "bold"),
        legend.title = element_blank(),
        legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(0, 100, by = 10)) +
  coord_cartesian(ylim = c(70, 101))  # Setting the visible area without dropping data
dev.off()


############################################################
### Fig S3 Full-text 

human_gpt <- read.csv(file = "Figure_Datasets/Table5-human_gpt_h2h.csv")
human_gpt$Accuracy <- as.numeric(human_gpt$Accuracy) * 100
human_gpt$Sensitivity <- as.numeric(human_gpt$Sensitivity) * 100

human_gpt <- human_gpt %>%
  mutate(
    SE_Sensitivity = sqrt((Sensitivity/100) * (1 - (Sensitivity/100)) / (TP+FN)) * 100,
    SE_Accuracy = sqrt((Accuracy/100) * (1 - (Accuracy/100)) / (TP+FP+TN+FN)) * 100)

plot_data <- human_gpt %>%
  pivot_longer(cols = c("Accuracy", "Sensitivity"), names_to = "Metric", values_to = "Value")
plot_data$Metric <- factor(plot_data$Metric, levels = c("Sensitivity", "Accuracy"))
plot_data <- plot_data %>%
  mutate(SE = if_else(Metric == "Sensitivity", SE_Sensitivity, SE_Accuracy))

pdf("supp_fig/figs3/ft_bar.pdf", width = 5, height = 7)
ggplot(plot_data, aes(x = Dataset, y = Value, fill = Prompt)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.7) +
  geom_errorbar(aes(ymin = Value - 1.96 * SE, ymax = pmin(Value + 1.96 * SE, 100)),
                position=position_dodge(width = 0.7),
                width=0.25,
                color = "#707070") + 
  scale_fill_manual(values = c("Human" = "#DB4345", "ISO-ScreenPrompt" = "#3C5488")) +
  labs(x = "Dataset", y = "Percentage (%)", title = "Dual Human vs. ISO-ScreenPrompt (Full-text)") +
  facet_wrap(~ Metric, scales = "free", ncol = 1) +
  theme_minimal() +
  theme(axis.line = element_line(color = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"),
        axis.text.y = element_text(face = "bold"),
        strip.text.x = element_text(face = "bold"),
        legend.title = element_blank(),
        legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = 0), breaks = seq(40, 100, by = 20)) +
  coord_cartesian(ylim = c(40, 101))  # Setting the visible area without dropping data
dev.off()


## FIg S3 Forest Plot (Single Reviewer)

library(dplyr)
library(ggplot2)
library(dplyr)
library(ggplot2)
library(tidyr)

h2h_abstract <- read.csv(file = "Figure_Datasets/Table5-human_gpt_h2h_abs.csv")

plot_data <- h2h_abstract %>%
  group_by(Dataset) %>%
  summarise(
    Human_Sensitivity = Sensitivity[Prompt == "Human"],
    ScreenPrompt_Sensitivity = Sensitivity[Prompt == "ScreenPrompt"],
    Human_Accuracy = Accuracy[Prompt == "Human"],
    ScreenPrompt_Accuracy = Accuracy[Prompt == "ScreenPrompt"],
    n_Human_Sens = TP[Prompt == "Human"] + FN[Prompt == "Human"],  
    n_ScreenPrompt_Sens = TP[Prompt == "ScreenPrompt"] + FN[Prompt == "ScreenPrompt"],  # Same correction here
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

View(plot_data)
pdf("supp_fig/figs3/figs3_forest_single_abs.pdf", width = 7, height = 7)
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