extern crate rand;
extern crate getopts;

use rand::sample;
use std::ops::{Add, Sub};
use std::iter::IntoIterator;
use std::io;
use getopts::Options;
use std::env;

fn print_usage(program: &str, opts: &Options) -> String {

	let brief = format!("Usage: {} FILE [options]", program);

	opts.usage(&brief)
}

fn do_randomizer() -> Result<Vec<(String, String)>, String> {

	let args: Vec<String> = env::args().collect();
	let program = args[0].clone();

	let mut opts = Options::new();

	let lower_sym = "l";
	let upper_sym = "u";
	let width_sym = "w";
	let fp_sym = "f";
	let help_sym = "h";

	opts.optopt(&lower_sym, "lower", "Lower bound in the range", "LOWER_BOUND");
	opts.optopt(&upper_sym, "upper", "Upper bound in the range", "UPPER_BOUND");
	opts.optopt(&width_sym, "with", "Interval width between two nearest points on the range", "INTERVAL");

	opts.optflag(&fp_sym, "floating-point", "Floating point mode");
	opts.optflag(&help_sym, "help", "print this help menu");
	
	let matches = try!(opts.parse(&args[1..]).map_err(|_| print_usage(&program, &opts)));

	if matches.opt_present(&help_sym) {
		return Err(print_usage(&program, &opts));
	}

	let lb = try!(matches.opt_str(&lower_sym).ok_or(print_usage(&program, &opts)));

	let ub = try!(matches.opt_str(&upper_sym).ok_or(print_usage(&program, &opts)));

	let iw = try!(matches.opt_str(&width_sym).ok_or(print_usage(&program, &opts)));
	
	let is_fp = matches.opt_present(&fp_sym);
	

	let mut input = String::new();

	let mut persons = Vec::new();

	while let Ok(n) = io::stdin().read_line(&mut input) {

		if n == 0 {
			break;
		}
		
		persons.push(String::from(input.trim()));
		input.clear();
	}

	fn parse_param<T>(lb: String, ub: String, iw: String) -> Result<(T, T, T), String> where T: std::str::FromStr + PartialOrd {
		let lb_v = try!(lb.parse::<T>().map_err(|_| "Error: Lower bound value is not number.\n".to_owned()));

		let ub_v = try!(ub.parse::<T>().map_err(|_| "Error: Upper bound value is not number.\n".to_owned()));

		let iw_v = try!(iw.parse::<T>().map_err(|_| "Error: Pointwise width value is not number\n".to_owned()));

		if lb_v > ub_v {
			return Err("Error: Upper bound is larger than or equal to Lower bound.".to_owned());
		}

		Ok((lb_v, ub_v, iw_v))
	}

	fn output_result<T>(persons: Vec<String>, samples: Vec<T>) -> Result<Vec<(String, String)>, String> where T: std::fmt::Display {

		Ok(persons.iter().enumerate().map(|x| (x.1.to_string(), samples.get(x.0).unwrap().to_string())).collect::<Vec<(String, String)>>())
	}

	if is_fp {

		let (lb_f, ub_f, iw_f) = try!(parse_param::<f64>(lb, ub, iw).map_err(|x| format!("{}\n{}", x, print_usage(&program, &opts))));

		let t = TwoPointsInterval::new(lb_f, ub_f, 0f64, iw_f);

		let samples = sample_at_random(t, persons.len())
				.iter()
				.map(|&x| (((x as f64) + 0.001f64) * 100f64).floor() / 100f64)
				.collect::<Vec<f64>>();

		output_result(persons, samples)
		
	} else {

		let (lb_i, ub_i, iw_i) = try!(parse_param::<i32>(lb, ub, iw).map_err(|x| format!("{}\n{}", x, print_usage(&program, &opts))));

		let t = TwoPointsInterval::new(lb_i, ub_i, 0i32, iw_i);

		let samples = sample_at_random(t, persons.len());
		
		output_result(persons, samples)
	}

}

fn main() {

	match do_randomizer() {
		Ok(k) => {
			for (name, sample) in k {
				println!("{}: {}", name, sample);
			}
		},
		Err(e) => {
			eprint!("{}", e);
		},
	}

}


fn sample_at_random<T, U>(domain: U, amount: usize) -> Vec<T> where T: PartialOrd + Add<Output = T> + Sub<Output = T> + Clone, U: IntoIterator<Item = T> {

	let mut rng = rand::thread_rng();

	sample(&mut rng, domain, amount)
}


#[derive(Clone)]
struct TwoPointsInterval<T> where T: Clone {

	start: T,
	end: T,
	zero_point: T,
	pointwise_width: T,
}

struct TwoPointsIntervalIterator<T> where T: Clone {

	interval: TwoPointsInterval<T>,
	current: T,
	is_terminal: bool,
}

impl<T> TwoPointsIntervalIterator<T> where T: Clone {

	fn new(interval: TwoPointsInterval<T>) -> Self {

		let current = interval.start.clone();

		TwoPointsIntervalIterator { interval, current, is_terminal: false }
	}
}

impl<T> TwoPointsInterval<T> where T: Clone {

	fn new(start: T, end: T, zero_point: T, pointwise_width: T) -> Self {

		TwoPointsInterval { start, end, zero_point, pointwise_width }
	}
}

impl<T> IntoIterator for TwoPointsInterval<T> where T: Add<Output = T> + Sub<Output = T> + PartialOrd + Clone {
	type Item = T;
	type IntoIter = TwoPointsIntervalIterator<T>;

	fn into_iter(self) -> Self::IntoIter {
		TwoPointsIntervalIterator::new(self.clone())
	}
}

impl<T> Iterator for TwoPointsIntervalIterator<T> where T: Add<Output = T> + Sub<Output = T> + PartialOrd + Clone {
	type Item = T;

	fn next(&mut self) -> Option<T> {

		if self.current < self.interval.end {

			let c = self.current.clone();
			let next_current = self.current.clone();

			self.current = next_current + self.interval.pointwise_width.clone();

			Some(c)

		} else {

			let diff = self.interval.end.clone() - self.current.clone();

			if ! self.is_terminal && (diff < self.interval.pointwise_width && diff >= self.interval.zero_point) {

				self.is_terminal = true;

				self.current = self.interval.end.clone();
				
				Some(self.current.clone())

			} else {

				None
			}
		}
	}
	
}

#[test]
fn test_two_points_interval() {

	let step = 0.5;

	let init_start = 1f32;

	let init_end = 4f32;

	let test_interval = TwoPointsInterval::new(init_start.clone(), init_end, 0.0, *&step);

	let mut v_per_step = init_start;

	for v in test_interval {

		assert_eq!(v, v_per_step);

		v_per_step = v_per_step + step;
	}

}

#[test]
fn test_two_points_interval_counter() {

	let step = 0.5;

	let init_start = 1f32;

	let init_end = 4f32;

	let loop_cnt = 1usize + ((init_end - init_start) / step) as usize;

	let test_interval = TwoPointsInterval::new(init_start.clone(), 4f32, 0.0, *&step);

	let mut i = 0;

	for _ in test_interval {

		i = i + 1;
	}

	assert_eq!(loop_cnt, i);

}

#[test]
fn test_two_points_interval_generation() {

	let test_interval = TwoPointsInterval::new(10f64, 20f64, 0.0, 0.1);

	let sampling_num = 50;

	for _ in 0..100 {

		let sample = sample_at_random(test_interval.clone(), *&sampling_num);

		let mut norm = sample.iter().map(|&x| ((x + 0.001f64) * 100f64).floor()/100f64).collect::<Vec<f64>>();

		norm.dedup();

		assert_eq!(norm.len(), sampling_num)
	}
}

#[test]
fn test_two_points_interval_generation_integer() {

	let test_interval = TwoPointsInterval::new(10, 100, 0, 1);

	let sampling_num = 50;

	for _ in 0..100 {

		let mut sample = sample_at_random(test_interval.clone(), *&sampling_num);

		sample.dedup();

		assert_eq!(sample.len(), sampling_num);
	}
}
