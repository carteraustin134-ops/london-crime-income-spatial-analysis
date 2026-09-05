# =============================================================
# The Relationship Between Income and Violent Crime Across
# London's Neighbourhoods
#
# A spatial regression analysis of socioeconomic predictors of
# violent crime across 4,835 London LSOAs (2018-2019).
# =============================================================

# 1. LOAD PACKAGES ---------------------------------------------
chooseCRANmirror(ind = 67)
packages <- c(
  "tidyverse",
  "sf",
  "tmap",
  "sfdep",
  "spdep",
  "lmtest",
  "car",
  "dbscan",
  "spatialreg",
  "gridExtra"
)
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# 2. LOAD DATA ---------------------------------------------------
# Load the three CSV datasets
crime <- read_csv("Crime Data/LONDON_CRIME_RATE_2019.csv")
lsoa_data <- read_csv("Crime Data/LSOA_DATA.csv")
deprivation <- read_csv("Crime Data/LONDON_DEPRIVATION_2019.csv")

# Load the shapefile
lsoa_shape <- st_read("Crime Data/LSOA shapefile/LSOA.shp")

# Check dimensions
dim(crime)
dim(lsoa_data)
dim(deprivation)
nrow(lsoa_shape)

# 3. MERGE DATASETS ------------------------------------------------
# Merge all three CSVs by LSOA code
data <- crime %>%
  left_join(lsoa_data, by = "LSOA") %>%
  left_join(deprivation, by = "LSOA")

# Check merge was successful
dim(data)

# 4. DESCRIPTIVE STATISTICS -----------------------------------------
# Summary statistics for key variables
summary(data[, c("RATE_per_1000", "MEAN_INCOME_2013", "INCOME_RANK", "Age_16_23")])

# Standard deviations
sapply(data[, c("RATE_per_1000", "MEAN_INCOME_2013", "INCOME_RANK", "Age_16_23")],
       sd, na.rm = TRUE)

# 5. LOG TRANSFORMATION OF DEPENDENT VARIABLE -----------------------
# Create log transformed crime rate variable
data$log_crime <- log(data$RATE_per_1000)

# Summary statistics including log transformed variable
summary(data[, c("RATE_per_1000", "log_crime", "MEAN_INCOME_2013", "INCOME_RANK",
                 "Age_16_23")])
sapply(data[, c("RATE_per_1000", "log_crime", "MEAN_INCOME_2013", "INCOME_RANK",
                "Age_16_23")],
       sd, na.rm = TRUE)

# 6. FIGURE 2 - HISTOGRAMS (RAW AND LOG TRANSFORMED) -----------------
hist1 <- ggplot(data, aes(x = RATE_per_1000)) +
  geom_histogram(binwidth = 5, fill = "skyblue", color = "black") +
  labs(
    title = "A)",
    x = "Crime Rate\n(per 1,000 people)",
    y = "Count"
  ) +
  coord_cartesian(xlim = c(0, 100)) +
  theme_minimal(base_size = 16)

hist2 <- ggplot(data, aes(x = log_crime)) +
  geom_histogram(binwidth = 0.2, fill = "skyblue", color = "black") +
  labs(
    title = "B)",
    x = "Log Crime Rate\n(per 1,000 people)",
    y = "Count"
  ) +
  theme_minimal(base_size = 16)

grid.arrange(hist1, hist2, ncol = 2)

# 7. CORRELATION ANALYSIS ---------------------------------------------
# Pearson correlation matrix
cor(data[, c("log_crime", "MEAN_INCOME_2013", "INCOME_RANK", "Age_16_23")],
    use = "complete.obs")

# 8. FIGURE 3 - SCATTER PLOTS -------------------------------------------
scatter1 <- ggplot(data, aes(x = MEAN_INCOME_2013, y = log_crime)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "lm", col = "red") +
  labs(
    title = "A)",
    x = "Mean Income (£)",
    y = "Log Crime Rate"
  ) +
  theme_minimal(base_size = 16) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

scatter2 <- ggplot(data, aes(x = INCOME_RANK, y = log_crime)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "lm", col = "red") +
  labs(
    title = "B)",
    x = "Deprivation Rank",
    y = "Log Crime Rate"
  ) +
  theme_minimal(base_size = 16) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

scatter3 <- ggplot(data, aes(x = Age_16_23, y = log_crime)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_smooth(method = "lm", col = "red") +
  labs(
    title = "C)",
    x = "% Aged 16-23",
    y = "Log Crime Rate"
  ) +
  theme_minimal(base_size = 16)

grid.arrange(scatter1, scatter2, scatter3, ncol = 3)

# 9. OLS REGRESSION MODEL -------------------------------------------------
# Fit the OLS regression model
model_ols <- lm(log_crime ~ MEAN_INCOME_2013 + INCOME_RANK + Age_16_23,
                 data = data)

# View model summary
summary(model_ols)

# Check for multicollinearity using VIF
vif(model_ols)

# Check BLUE assumptions using diagnostic plots
par(mfrow = c(2,2))
plot(model_ols)

# Breusch-Pagan test for heteroscedasticity
bptest(model_ols)

# 10. SPATIAL DATA PREPARATION ---------------------------------------------
# Join data to shapefile
data_spatial <- lsoa_shape %>%
  left_join(data, by = "LSOA")

# Add OLS residuals to spatial data
data_spatial$ols_resid <- residuals(model_ols)

# Convert polygons to centroids for spatial analysis
data_spatial_pts <- st_centroid(data_spatial)

# 11. ELBOW PLOT FOR SELECTING k NEAREST NEIGHBOURS -------------------------
coords <- st_coordinates(data_spatial_pts)
k_search <- kNN(coords, k = 300)
nn_matrix <- k_search$id
x <- data_spatial_pts$ols_resid
neighbour_vals <- matrix(x[nn_matrix], ncol = 300)
r_values <- as.vector(cor(x, neighbour_vals))
elbow_data <- data.frame(k = 1:300, r = r_values)

ggplot(elbow_data, aes(x = k, y = r)) +
  geom_line() +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = mean(elbow_data$r), linetype = "dashed", colour = "red") +
  theme_minimal() +
  labs(
    title = "Elbow Plot for Selecting k Nearest Neighbours",
    x = "k (number of nearest neighbours)",
    y = "Correlation (r)"
  )

# 12. SPATIAL WEIGHTS AND MORAN'S I -----------------------------------------
# Calculate spatial weights using k = 100 nearest neighbours
neighbours <- st_knn(geometry = data_spatial_pts, k = 100)
weights <- st_weights(nb = neighbours, allow_zero = TRUE)

# Global Moran's I test on OLS residuals
sfdep::global_moran_test(x = data_spatial_pts$ols_resid, nb = neighbours, wt = weights)

# 13. FIGURE 4 - MORAN'S I SCATTERPLOT --------------------------------------
data_spatial_pts$lag_resid <- st_lag(x = data_spatial_pts$ols_resid,
                                      nb = neighbours,
                                      wt = weights,
                                      allow_zero = TRUE)

ggplot(data_spatial_pts, aes(x = ols_resid, y = lag_resid)) +
  geom_point(pch = 3) +
  geom_smooth(method = "lm", se = FALSE, col = "red", lwd = 0.2, fullrange = TRUE) +
  geom_vline(xintercept = 0, linetype = "dashed", col = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", col = "blue") +
  labs(
    title = "Moran's I Scatterplot",
    x = "OLS Residuals",
    y = "Spatially Lagged Residuals"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# 14. SPATIAL LAG MODEL ------------------------------------------------------
# Fit the spatial lag model
model_spatial <- spatialreg::lagsarlm(
  log_crime ~ MEAN_INCOME_2013 + INCOME_RANK + Age_16_23,
  data = data_spatial,
  listw = spdep::nb2listw(neighbours, style = "W")
)

# View model summary
summary(model_spatial)

# Add spatial model residuals to spatial data
data_spatial$spatial_resid <- residuals(model_spatial)

# 15. FIGURES 1 AND 5 - MAPS -------------------------------------------------
# Transform to WGS84 for mapping
data_spatial_wgs84 <- st_transform(data_spatial, 4326)

# Figure 1: Map of violent crime rate
map1 <- tm_shape(data_spatial_wgs84) +
  tm_polygons(
    fill = "RATE_per_1000",
    fill.scale = tm_scale_intervals(
      style = "quantile",
      values = "YlOrRd"
    ),
    fill.legend = tm_legend(title = "Violent Crime Rate\nper 1,000 people"),
    col_alpha = 0
  ) +
  tm_graticules() +
  tm_compass(position = c("right", "top")) +
  tm_scalebar(position = c("left", "bottom"))
map1

# Figure 5: Map of spatial lag model residuals
map5 <- tm_shape(data_spatial_wgs84) +
  tm_polygons(
    fill = "spatial_resid",
    fill.scale = tm_scale_intervals(
      style = "quantile",
      values = "brewer.br_bg"
    ),
    fill.legend = tm_legend(title = "Spatial Model\nResiduals"),
    col_alpha = 0
  ) +
  tm_graticules() +
  tm_compass(position = c("right", "top")) +
  tm_scalebar(position = c("left", "bottom"))
map5
