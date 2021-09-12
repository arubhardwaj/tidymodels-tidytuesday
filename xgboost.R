library(xgboost)
library(tidyverse)

vb_matches <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-19/vb_matches.csv', guess_max = 76000)
vb_matches
#beepr::beep()


vb_parsed <- vb_matches %>% transmute(circuit,
                         gender,
                         year,
                         w_attacks = w_p1_tot_attacks+w_p2_tot_attacks,
                         w_kills = w_p1_tot_kills+w_p2_tot_kills,
                         w_errors = w_p1_tot_errors+w_p2_tot_errors,
                         w_aces = w_p1_tot_aces+w_p2_tot_aces,
                         w_serve_errors = w_p1_tot_serve_errors+w_p2_tot_serve_errors,
                         w_digs = w_p1_tot_digs+w_p2_tot_digs,
                         l_attacks = l_p1_tot_attacks+l_p2_tot_attacks,
                         l_kills = l_p1_tot_kills+l_p2_tot_kills,
                         l_errors = l_p1_tot_errors+l_p2_tot_errors,
                         l_aces = l_p1_tot_aces+l_p2_tot_aces,
                         l_serve_errors = l_p1_tot_serve_errors+l_p2_tot_serve_errors,
                         l_digs = l_p1_tot_digs+l_p2_tot_digs) %>% 
  na.omit()


# Wrangling

winners <- vb_parsed %>% select(circuit, gender, year,
                     w_attacks:w_digs) %>% 
  rename_with(~ str_remove_all(.,"w_"), w_attacks:w_digs) %>% 
  mutate(win="win")


losers <- vb_parsed %>% select(circuit, gender, year,
                                l_attacks:l_digs) %>% 
  rename_with(~ str_remove_all(.,"l_"), l_attacks:l_digs) %>% 
  mutate(win="loose")

vb_df <- bind_rows(winners, losers) %>% 
  mutate_if(is.character, factor)



vb_df %>% 
  pivot_longer(attacks:digs, names_to = "stat", values_to = "value") %>% 
  ggplot(aes(gender,value, fill=win, color = win))+
  geom_boxplot(alpha = 0.4)+
  facet_wrap(~stat, scales = "free_y", nrow =2)+
  labs(y=NULL, color = NULL, fill = NULL)



# Model -------------------------------------------------------------------

library(tidymodels)
vb_split <- initial_split(vb_df, starta = win)
vb_train <- training(vb_split)
vb_test <- testing(vb_split)

xgb_spec <- boost_tree(
  trees = 1000,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(),
  sample_size = tune(), mtry=tune(), learn_rate = tune()
) %>% 
  set_engine("xgboost") %>% set_mode("classification")

xgb_spec


xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), vb_train),
  learn_rate(),
  size=20
)

xgb_grid


xgb_wf <- workflow() %>% 
  add_formula(win~.) %>% 
  add_model(xgb_spec)



set.seed(123)

vb_folds <- vfold_cv(vb_train,starta=win)
vb_folds



doParallel::registerDoParallel()

set.seed(1234)

xgb_res <- tune_grid(
  xgb_wf,
  resamples = vb_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = T)
)


beepr::beep()


# Explore Models

xgb_res %>% collect_metrics() %>% 
  filter(.metric=="roc_auc") %>% select(mean, mtry:sample_size) %>% 
  pivot_longer(mtry:sample_size, names_to = "parameter",
               values_to = "value") %>% 
  ggplot(aes(value, mean, col = parameter))+
  geom_point(show.legend = F, size = 2)+
  facet_wrap(~parameter, scales = "free_x")


show_best(xgb_res, "roc_auc")


best_auc<-select_best(xgb_res, "roc_auc")

final_xgb <- finalize_workflow(xgb_wf, best_auc)


library(vip)


final_xgb %>% fit(data=vb_train) %>% 
  pull_workflow_fit() %>% 
  vip(geom="point")

final_res <- last_fit(final_xgb, vb_split) 
final_res %>% collect_metrics()

final_res %>% collect_predictions() %>% roc_curve(win,.pred_win) %>% 
  autoplot()
