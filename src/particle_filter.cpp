/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_PARTICLES 100
#define YAW_RATE_THRESHOLD 0.001

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set the number of particles.
	num_particles = NUM_PARTICLES;

	// Initialize particles to first position
	normal_distribution<double> noise_x_init(x, std[0]);
	normal_distribution<double> noise_y_init(y, std[1]);
	normal_distribution<double> noise_theta_init(theta, std[2]);

	for(unsigned int i = 0; i < num_particles; i++) {
		Particle p;
		p.id     = i;
		p.weight = 1.0f;
		// Adding noise
		p.x      = noise_x_init(gen);
		p.y      = noise_y_init(gen);
		p.theta  = noise_theta_init(gen);
		// Appending particle object to particle vector
		particles.push_back(p);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> noise_x(0., std_pos[0]);
	normal_distribution<double> noise_y(0., std_pos[1]);
	normal_distribution<double> noise_theta(0., std_pos[2]);

	double yaw_rate_dt  = yaw_rate * delta_t;
	double vel_yaw_rate = velocity / yaw_rate;
	double vel_dt       = velocity * delta_t;

	for (unsigned int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < YAW_RATE_THRESHOLD) {  
			particles[i].x     += vel_dt * cos(particles[i].theta);
			particles[i].y     += vel_dt * sin(particles[i].theta);
		} 
		else {
			particles[i].x     += vel_yaw_rate * (sin(particles[i].theta + yaw_rate_dt) - sin(particles[i].theta));
			particles[i].y     += vel_yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_dt));
			particles[i].theta += yaw_rate_dt;
		}
		// Adding Gaussian noise
		particles[i].x     += noise_x(gen);
		particles[i].y     += noise_y(gen);
		particles[i].theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); i++) {
		LandmarkObs observation = observations[i];

		// Initialize the min distance to highest value and map_id
		double min_dist = numeric_limits<double>::max();
		int map_id      = -1;

		for (unsigned int j = 0; j < predicted.size(); j++) {
			LandmarkObs prediction = predicted[j];
			double current_dist    = dist(observation.x, observation.y, prediction.x, prediction.y);

			// Find nearest landmark
			if (current_dist < min_dist) {
				min_dist = current_dist;
				map_id   = prediction.id;
			}
		}
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for(int i = 0; i < num_particles; i++) {

		double p_x     = particles[i].x;
		double p_y     = particles[i].y;
		double p_theta = particles[i].theta;
		
		vector<LandmarkObs> predicted_landmarks;
		
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			double l_x = map_landmarks.landmark_list[j].x_f;
			double l_y = map_landmarks.landmark_list[j].y_f;
			int l_id   = map_landmarks.landmark_list[j].id_i;
			// Check for observations within sensor range
			if (fabs(l_x - p_x) <= sensor_range && fabs(l_y - p_y) <= sensor_range) {
				predicted_landmarks.push_back(LandmarkObs{ l_id, l_x, l_y });
			}
		}

		vector<LandmarkObs> transformed_observations;
		
		// pre-computing reusable values
		double sin_theta = sin(p_theta);
		double cos_theta = cos(p_theta);
		
		for (unsigned int j = 0; j < observations.size(); j++) {
			double transformed_x = cos_theta * observations[j].x - sin_theta * observations[j].y + p_x;
			double transformed_y = sin_theta * observations[j].x + cos_theta * observations[j].y + p_y;
			
			transformed_observations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
		}

		dataAssociation(predicted_landmarks, transformed_observations);

		// re-initializing weights
		particles[i].weight = 1.0;
		
		// pre-computing reusable values
		double std_x      = std_landmark[0];
		double std_y      = std_landmark[1];
		double normalizer = (1 / (2 * M_PI * std_x * std_y));

		for (unsigned int j = 0; j < transformed_observations.size(); j++) {

			double obs_x = transformed_observations[j].x;
			double obs_y = transformed_observations[j].y;

			double pred_x, pred_y;

			int associated_landmark_id = transformed_observations[j].id;
			
			// Looping through the list assuming it could be unsorted
			for (unsigned int j = 0; j < predicted_landmarks.size(); j++) {
				if (predicted_landmarks[j].id == associated_landmark_id) {	
					pred_x = predicted_landmarks[j].x;
					pred_y = predicted_landmarks[j].y;
				}
			}
			double observation_probability = normalizer * exp(-(pow(obs_x - pred_x, 2) / (2 * pow(std_x, 2)) + (pow(obs_y - pred_y, 2) / (2 * pow(std_y, 2)))));
			
			particles[i].weight *= observation_probability;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> new_particles;

	vector<double> weights;
	for (int x = 0; x < num_particles; x++)
	{
		weights.push_back(particles[x].weight);
	}

	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	auto index = uniintdist(gen);

	// Get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// Uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	// Spin the resample wheel
	for (int x = 0; x < num_particles; x++)
	{
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index])
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
