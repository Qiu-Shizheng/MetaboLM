#############################################
# 1. Load Necessary Packages
#############################################
library(survival)
library(timeROC)
library(dplyr)
library(foreach)
library(doParallel)

set.seed(123)

cl <- makeCluster(30)
registerDoParallel(cl)

#############################################
# 2. Read Data and Preprocess
#############################################
# Read baseline data
df <- read.csv("your/path/baseline.csv", stringsAsFactors = FALSE)

# Read healthy control IDs
healthy_ids <- read.csv("your/path/healthy_eids.csv", stringsAsFactors = FALSE)$eid

# Convert key date variables:
# p200: Baseline date; p40000_i0: Death date
df$p200 <- as.Date(df$p200)
df$p40000_i0 <- as.Date(df$p40000_i0)

# Define disease list and their corresponding column names in the data
disease_list <- list(
  asthma = "p131494",
  dementia = "p42018",
  copd = "p42016",
  stroke = "p42006",
  parkinson = "p42032",
  hypertension = "p131286",
  ischaemic_heart_disease = "p131306",
  atrial_fibrillation = "p131350",
  heart_failure = "p131354",
  obesity = "p130792",
  diabetes = "p130708",
  RA = "RA",
  Colorectal_cancer = "Colorectal_cancer",
  Lung_cancer = "Lung_cancer",
  Breast_cancer = "Breast_cancer",
  Prostate_cancer = "Prostate_cancer"
)

# Convert the diagnosis date for each disease (if available) to Date type
for(col in disease_list) {
  if(col %in% names(df)) {
    df[[col]] <- as.Date(df[[col]])
  } else {
    warning(paste("Column", col, "does not exist in the data, skipping date conversion."))
  }
}

# Set the follow-up cutoff date
cutoff_date <- as.Date("2024-09-01")

#############################################
# 3. Define Helper Functions
#############################################

# 3.1 Compute Follow-up Time
compute_followup <- function(baseline, event_date, death_date, cutoff) {
  event_time <- ifelse(!is.na(event_date), as.numeric(event_date - baseline), NA)
  
  # Preserve date class properties
  censor_date <- rep(cutoff, length(baseline))
  non_na_death <- !is.na(death_date)
  censor_date[non_na_death] <- pmin(death_date[non_na_death], cutoff)
  
  censor_time <- as.numeric(censor_date - baseline)
  result <- ifelse(!is.na(event_time), event_time, censor_time)
  result_years <- result / 365.25
  return(result_years)
}

# 3.2 Compute Group Metrics
compute_group_metrics <- function(data, risk_score, start_time, end_time) {
  if (is.finite(end_time)) {
    idx <- which(data$time > start_time)
    subset_data <- data[idx, ]
    if (nrow(subset_data) == 0) {
      cat("No samples in the group between start_time =", start_time, "and end_time =", end_time, "\n")
      return(list(AUC = NA, cindex = NA, cindex_ci = c(lower = NA, upper = NA)))
    }
    new_time <- pmin(subset_data$time - start_time, end_time - start_time)
    new_event <- ifelse(subset_data$time <= end_time & subset_data$event == 1, 1, 0)
    horizon <- end_time - start_time
    risk_used <- risk_score[idx]
  } else {
    subset_data <- data
    new_time <- subset_data$time
    new_event <- subset_data$event
    horizon <- as.numeric(quantile(new_time, probs = 0.95, na.rm = TRUE))
    risk_used <- risk_score
  }
  
  valid <- !is.na(new_time) & !is.na(new_event) & !is.na(risk_used)
  new_time <- new_time[valid]
  new_event <- new_event[valid]
  risk_used <- risk_used[valid]
  if (length(new_time) == 0) {
    cat("In group with start_time =", start_time, "no samples remain after removing missing values\n")
    return(list(AUC = NA, cindex = NA, cindex_ci = c(lower = NA, upper = NA)))
  }
  
  cat("Group: start_time =", start_time, 
      "end_time =", if (is.finite(end_time)) end_time else "All", "\n")
  cat("  Sample size:", length(new_time), "\n")
  cat("  Number of events:", sum(new_event), "\n")
  
  # Calculate time-dependent AUC
  auc_val <- tryCatch({
    max_event_time <- max(new_time[new_event == 1], na.rm = TRUE)
    horizon_eff <- if(horizon > max_event_time) max_event_time else horizon
    tmp <- timeROC(T = new_time,
                   delta = new_event,
                   marker = risk_used,
                   cause = 1,
                   weighting = "marginal",
                   times = horizon_eff,
                   ROC = TRUE,
                   iid = FALSE)
    tmp$AUC[1]
  }, error = function(e) {
    cat("  timeROC error:", conditionMessage(e), "\n")
    NA
  })
  
  # Calculate C-index
  cindex_obj <- survConcordance(Surv(new_time, new_event) ~ risk_used)
  cindex_val <- cindex_obj$concordance
  
  # Compute 95% CI for the C-index using parallel bootstrap
  bootstrap_cindex_group <- function(df_sub, n_boot = 1000) {
    n <- nrow(df_sub)
    boot_vals <- foreach(i = 1:n_boot, .combine = c, .packages = "survival") %dopar% {
      boot_idx <- sample(1:n, size = n, replace = TRUE)
      boot_data <- df_sub[boot_idx, ]
      sc_boot <- tryCatch({
        survConcordance(Surv(boot_data$new_time, boot_data$new_event) ~ boot_data$risk_score)
      }, error = function(e) NULL)
      if (!is.null(sc_boot)) {
        sc_boot$concordance
      } else {
        NA
      }
    }
    boot_vals <- boot_vals[!is.na(boot_vals)]
    if (length(boot_vals) == 0)
      return(c(lower = NA, upper = NA))
    ci_lower <- unname(as.numeric(quantile(boot_vals, probs = 0.025)))
    ci_upper <- unname(as.numeric(quantile(boot_vals, probs = 0.975)))
    return(c(lower = ci_lower, upper = ci_upper))
  }
  
  df_boot <- data.frame(new_time = new_time,
                        new_event = new_event,
                        risk_score = risk_used)
  boot_ci <- bootstrap_cindex_group(df_boot, n_boot = 1000)
  
  return(list(AUC = auc_val, cindex = cindex_val, cindex_ci = boot_ci))
}

#############################################
# 4. Build the Model and Evaluate Groups
#############################################
# Create a results dataframe
results <- data.frame(
  Disease = character(),
  Group = character(), 
  AUC = numeric(),
  C_index = numeric(),
  C_index_CI_lower = numeric(),
  C_index_CI_upper = numeric(),
  stringsAsFactors = FALSE
)

# Define time groups: 0-10 years, 10-20 years, All follow-up
time_groups <- list("Within 10 years" = c(0, 10),
                    "10-20 years" = c(10, 20),
                    "All follow-up" = c(0, Inf))

total_tasks <- length(disease_list) * length(time_groups)
completed_tasks <- 0
pb <- txtProgressBar(min = 0, max = total_tasks, style = 3)

for (disease in names(disease_list)) {
  cat("============================================\n")
  cat("Processing disease:", disease, "\n")
  
  disease_col <- disease_list[[disease]]
  
  df_disease <- df %>%
    filter((eid %in% healthy_ids) | (!is.na(.[[disease_col]]))) %>%
    filter(is.na(.[[disease_col]]) | (.[[disease_col]] >= p200))
  
  cat("After filtering, disease", disease, "record count:", nrow(df_disease), "\n")
  
  # Create event variable: 1 if record exists, 0 otherwise
  df_disease$event <- ifelse(!is.na(df_disease[[disease_col]]), 1, 0)
  cat("Disease", disease, "number of events:", sum(df_disease$event), "\n")
  
  # Compute follow-up time
  df_disease$time <- mapply(compute_followup,
                            baseline = df_disease$p200,
                            event_date = df_disease[[disease_col]],
                            death_date = df_disease$p40000_i0,
                            MoreArgs = list(cutoff = cutoff_date))
  
  # Remove samples with non-positive follow-up time
  df_disease <- df_disease %>% filter(time > 0)
  cat("Remaining record count after removing non-positive follow-up time:", nrow(df_disease), "\n")
  
  ######################################################
  # Load Risk Score File
  ######################################################
  risk_file <- paste0("your/path/finetune_daixie_0313_168_20250312_002754/", 
                      disease, "/finetune_predictions_", disease, ".csv")
  
  if (!file.exists(risk_file)) {
    cat("Risk score file does not exist for disease", disease, "\n")
    next
  }
  risk_df <- read.csv(risk_file, stringsAsFactors = FALSE)
  
  # Merge risk scores into df_disease by eid
  # The risk score file must include: eid and Predicted_Probability (risk score between 0 and 1)
  df_disease <- merge(df_disease, risk_df[, c("eid", "Predicted_Probability")],
                      by = "eid", all.x = TRUE)
  # Rename risk score column and convert to numeric
  names(df_disease)[names(df_disease) == "Predicted_Probability"] <- "risk_score"
  df_disease$risk_score <- as.numeric(df_disease$risk_score)
  
  df_disease_model <- df_disease[complete.cases(df_disease[, "risk_score"]), ]
  cat("Number of complete records for Cox model:", nrow(df_disease_model), "\n")
  
  if(nrow(df_disease_model) == 0){
    cat("Disease", disease, "has too many missing risk scores, skipping this disease.\n")
    next
  }
  
  ######################################################
  # Build a Cox Model using Only the Risk Score
  ######################################################
  cox_formula <- as.formula("Surv(time, event) ~ risk_score")
  
  model <- tryCatch(coxph(cox_formula, data = df_disease_model), error = function(e) {
    cat("Error building Cox model for disease", disease, ":", conditionMessage(e), "\n")
    return(NULL)
  })
  if (is.null(model)) {
    cat("Disease", disease, "skipped due to model error.\n")
    next
  }
  
  model_summary <- summary(model)
  global_cindex <- model_summary$concordance[1]
  cat("Disease", disease, "full follow-up model C-index =", round(global_cindex, 4), "\n")
  
  lp <- predict(model, type = "lp")
  if (length(lp) != nrow(df_disease_model)) {
    cat("Length of risk score does not match number of rows for disease", disease, ", skipping this disease.\n")
    next
  }
  
  # Evaluate for each time group
  for (group_name in names(time_groups)) {
    grp <- time_groups[[group_name]]  # grp[1]: start_time, grp[2]: end_time (or Inf for full follow-up)
    res_grp <- compute_group_metrics(df_disease_model, lp, start_time = grp[1], end_time = grp[2])
    
    cat("Disease:", disease, "Group:", group_name, 
        "AUC =", round(res_grp$AUC, 4), 
        "C-index =", round(res_grp$cindex, 4), "\n")
    
    results <- rbind(results, data.frame(
      Disease = disease,
      Group = group_name,
      AUC = res_grp$AUC,
      C_index = res_grp$cindex,
      C_index_CI_lower = res_grp$cindex_ci["lower"],
      C_index_CI_upper = res_grp$cindex_ci["upper"],
      stringsAsFactors = FALSE
    ))
    
    # Update progress bar
    completed_tasks <- completed_tasks + 1
    setTxtProgressBar(pb, completed_tasks)
  }
}

write.csv(results, file = "your/path/cox_model_results.csv", row.names = FALSE)
cat("\nAll model results have been saved to: cox_model_results.csv\n")

close(pb)
stopCluster(cl)#############################################
# 1. Load Necessary Packages
#############################################
library(survival)
library(timeROC)
library(dplyr)
library(foreach)
library(doParallel)

set.seed(123)

cl <- makeCluster(30)
registerDoParallel(cl)

#############################################
# 2. Read Data and Preprocess
#############################################
# Read baseline data
df <- read.csv("your/path/baseline.csv", stringsAsFactors = FALSE)

# Read healthy control IDs
healthy_ids <- read.csv("your/path/healthy_eids.csv", stringsAsFactors = FALSE)$eid

# Convert key date variables:
# p200: Baseline date; p40000_i0: Death date
df$p200 <- as.Date(df$p200)
df$p40000_i0 <- as.Date(df$p40000_i0)

# Define disease list and their corresponding column names in the data
disease_list <- list(
  asthma = "p131494",
  dementia = "p42018",
  copd = "p42016",
  stroke = "p42006",
  parkinson = "p42032",
  hypertension = "p131286",
  ischaemic_heart_disease = "p131306",
  atrial_fibrillation = "p131350",
  heart_failure = "p131354",
  obesity = "p130792",
  diabetes = "p130708",
  RA = "RA",
  Colorectal_cancer = "Colorectal_cancer",
  Lung_cancer = "Lung_cancer",
  Breast_cancer = "Breast_cancer",
  Prostate_cancer = "Prostate_cancer"
)

# Convert the diagnosis date for each disease (if available) to Date type
for(col in disease_list) {
  if(col %in% names(df)) {
    df[[col]] <- as.Date(df[[col]])
  } else {
    warning(paste("Column", col, "does not exist in the data, skipping date conversion."))
  }
}

# Set the follow-up cutoff date
cutoff_date <- as.Date("2024-09-01")

#############################################
# 3. Define Helper Functions
#############################################

# 3.1 Compute Follow-up Time
compute_followup <- function(baseline, event_date, death_date, cutoff) {
  event_time <- ifelse(!is.na(event_date), as.numeric(event_date - baseline), NA)
  
  # Preserve date class properties
  censor_date <- rep(cutoff, length(baseline))
  non_na_death <- !is.na(death_date)
  censor_date[non_na_death] <- pmin(death_date[non_na_death], cutoff)
  
  censor_time <- as.numeric(censor_date - baseline)
  result <- ifelse(!is.na(event_time), event_time, censor_time)
  result_years <- result / 365.25
  return(result_years)
}

# 3.2 Compute Group Metrics
compute_group_metrics <- function(data, risk_score, start_time, end_time) {
  if (is.finite(end_time)) {
    idx <- which(data$time > start_time)
    subset_data <- data[idx, ]
    if (nrow(subset_data) == 0) {
      cat("No samples in the group between start_time =", start_time, "and end_time =", end_time, "\n")
      return(list(AUC = NA, cindex = NA, cindex_ci = c(lower = NA, upper = NA)))
    }
    new_time <- pmin(subset_data$time - start_time, end_time - start_time)
    new_event <- ifelse(subset_data$time <= end_time & subset_data$event == 1, 1, 0)
    horizon <- end_time - start_time
    risk_used <- risk_score[idx]
  } else {
    subset_data <- data
    new_time <- subset_data$time
    new_event <- subset_data$event
    horizon <- as.numeric(quantile(new_time, probs = 0.95, na.rm = TRUE))
    risk_used <- risk_score
  }
  
  valid <- !is.na(new_time) & !is.na(new_event) & !is.na(risk_used)
  new_time <- new_time[valid]
  new_event <- new_event[valid]
  risk_used <- risk_used[valid]
  if (length(new_time) == 0) {
    cat("In group with start_time =", start_time, "no samples remain after removing missing values\n")
    return(list(AUC = NA, cindex = NA, cindex_ci = c(lower = NA, upper = NA)))
  }
  
  cat("Group: start_time =", start_time, 
      "end_time =", if (is.finite(end_time)) end_time else "All", "\n")
  cat("  Sample size:", length(new_time), "\n")
  cat("  Number of events:", sum(new_event), "\n")
  
  # Calculate time-dependent AUC
  auc_val <- tryCatch({
    max_event_time <- max(new_time[new_event == 1], na.rm = TRUE)
    horizon_eff <- if(horizon > max_event_time) max_event_time else horizon
    tmp <- timeROC(T = new_time,
                   delta = new_event,
                   marker = risk_used,
                   cause = 1,
                   weighting = "marginal",
                   times = horizon_eff,
                   ROC = TRUE,
                   iid = FALSE)
    tmp$AUC[1]
  }, error = function(e) {
    cat("  timeROC error:", conditionMessage(e), "\n")
    NA
  })
  
  # Calculate C-index
  cindex_obj <- survConcordance(Surv(new_time, new_event) ~ risk_used)
  cindex_val <- cindex_obj$concordance
  
  # Compute 95% CI for the C-index using parallel bootstrap
  bootstrap_cindex_group <- function(df_sub, n_boot = 1000) {
    n <- nrow(df_sub)
    boot_vals <- foreach(i = 1:n_boot, .combine = c, .packages = "survival") %dopar% {
      boot_idx <- sample(1:n, size = n, replace = TRUE)
      boot_data <- df_sub[boot_idx, ]
      sc_boot <- tryCatch({
        survConcordance(Surv(boot_data$new_time, boot_data$new_event) ~ boot_data$risk_score)
      }, error = function(e) NULL)
      if (!is.null(sc_boot)) {
        sc_boot$concordance
      } else {
        NA
      }
    }
    boot_vals <- boot_vals[!is.na(boot_vals)]
    if (length(boot_vals) == 0)
      return(c(lower = NA, upper = NA))
    ci_lower <- unname(as.numeric(quantile(boot_vals, probs = 0.025)))
    ci_upper <- unname(as.numeric(quantile(boot_vals, probs = 0.975)))
    return(c(lower = ci_lower, upper = ci_upper))
  }
  
  df_boot <- data.frame(new_time = new_time,
                        new_event = new_event,
                        risk_score = risk_used)
  boot_ci <- bootstrap_cindex_group(df_boot, n_boot = 1000)
  
  return(list(AUC = auc_val, cindex = cindex_val, cindex_ci = boot_ci))
}

#############################################
# 4. Build the Model and Evaluate Groups
#############################################
# Create a results dataframe
results <- data.frame(
  Disease = character(),
  Group = character(), 
  AUC = numeric(),
  C_index = numeric(),
  C_index_CI_lower = numeric(),
  C_index_CI_upper = numeric(),
  stringsAsFactors = FALSE
)

# Define time groups: 0-10 years, 10-20 years, All follow-up
time_groups <- list("Within 10 years" = c(0, 10),
                    "10-20 years" = c(10, 20),
                    "All follow-up" = c(0, Inf))

total_tasks <- length(disease_list) * length(time_groups)
completed_tasks <- 0
pb <- txtProgressBar(min = 0, max = total_tasks, style = 3)

for (disease in names(disease_list)) {
  cat("============================================\n")
  cat("Processing disease:", disease, "\n")
  
  disease_col <- disease_list[[disease]]
  
  df_disease <- df %>%
    filter((eid %in% healthy_ids) | (!is.na(.[[disease_col]]))) %>%
    filter(is.na(.[[disease_col]]) | (.[[disease_col]] >= p200))
  
  cat("After filtering, disease", disease, "record count:", nrow(df_disease), "\n")
  
  # Create event variable: 1 if record exists, 0 otherwise
  df_disease$event <- ifelse(!is.na(df_disease[[disease_col]]), 1, 0)
  cat("Disease", disease, "number of events:", sum(df_disease$event), "\n")
  
  # Compute follow-up time
  df_disease$time <- mapply(compute_followup,
                            baseline = df_disease$p200,
                            event_date = df_disease[[disease_col]],
                            death_date = df_disease$p40000_i0,
                            MoreArgs = list(cutoff = cutoff_date))
  
  # Remove samples with non-positive follow-up time
  df_disease <- df_disease %>% filter(time > 0)
  cat("Remaining record count after removing non-positive follow-up time:", nrow(df_disease), "\n")
  
  ######################################################
  # Load Risk Score File
  ######################################################
  risk_file <- paste0("your/path/", 
                      disease, "/finetune_predictions_", disease, ".csv")
  
  if (!file.exists(risk_file)) {
    cat("Risk score file does not exist for disease", disease, "\n")
    next
  }
  risk_df <- read.csv(risk_file, stringsAsFactors = FALSE)
  
  # Merge risk scores into df_disease by eid
  # The risk score file must include: eid and Predicted_Probability (risk score between 0 and 1)
  df_disease <- merge(df_disease, risk_df[, c("eid", "Predicted_Probability")],
                      by = "eid", all.x = TRUE)
  # Rename risk score column and convert to numeric
  names(df_disease)[names(df_disease) == "Predicted_Probability"] <- "risk_score"
  df_disease$risk_score <- as.numeric(df_disease$risk_score)
  
  df_disease_model <- df_disease[complete.cases(df_disease[, "risk_score"]), ]
  cat("Number of complete records for Cox model:", nrow(df_disease_model), "\n")
  
  if(nrow(df_disease_model) == 0){
    cat("Disease", disease, "has too many missing risk scores, skipping this disease.\n")
    next
  }
  
  ######################################################
  # Build a Cox Model using Only the Risk Score
  ######################################################
  cox_formula <- as.formula("Surv(time, event) ~ risk_score")
  
  model <- tryCatch(coxph(cox_formula, data = df_disease_model), error = function(e) {
    cat("Error building Cox model for disease", disease, ":", conditionMessage(e), "\n")
    return(NULL)
  })
  if (is.null(model)) {
    cat("Disease", disease, "skipped due to model error.\n")
    next
  }
  
  model_summary <- summary(model)
  global_cindex <- model_summary$concordance[1]
  cat("Disease", disease, "full follow-up model C-index =", round(global_cindex, 4), "\n")
  
  lp <- predict(model, type = "lp")
  if (length(lp) != nrow(df_disease_model)) {
    cat("Length of risk score does not match number of rows for disease", disease, ", skipping this disease.\n")
    next
  }
  
  # Evaluate for each time group
  for (group_name in names(time_groups)) {
    grp <- time_groups[[group_name]]  # grp[1]: start_time, grp[2]: end_time (or Inf for full follow-up)
    res_grp <- compute_group_metrics(df_disease_model, lp, start_time = grp[1], end_time = grp[2])
    
    cat("Disease:", disease, "Group:", group_name, 
        "AUC =", round(res_grp$AUC, 4), 
        "C-index =", round(res_grp$cindex, 4), "\n")
    
    results <- rbind(results, data.frame(
      Disease = disease,
      Group = group_name,
      AUC = res_grp$AUC,
      C_index = res_grp$cindex,
      C_index_CI_lower = res_grp$cindex_ci["lower"],
      C_index_CI_upper = res_grp$cindex_ci["upper"],
      stringsAsFactors = FALSE
    ))
    
    # Update progress bar
    completed_tasks <- completed_tasks + 1
    setTxtProgressBar(pb, completed_tasks)
  }
}

write.csv(results, file = "your/path/cox_model_results.csv", row.names = FALSE)
cat("\nAll model results have been saved to: cox_model_results.csv\n")

close(pb)

stopCluster(cl)
