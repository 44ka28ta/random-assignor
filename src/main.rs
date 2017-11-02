extern crate rand;
extern crate getopts;

use rand::sample;
use std::ops::{Add, Sub};
use std::iter::IntoIterator;
use std::io;
use getopts::Options;
use std::env;

fn print_usage(program: &str, opts: Options) {
	let brief = format!("Usage: {} FILE [options]", program);
	print!("{}", opts.usage(&brief));
}

fn main() {

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
	
	let matches = match opts.parse(&args[1..]) {
		Ok(m) => { m },
		Err(f) => { panic!(f.to_string()) }
	};

	if matches.opt_present(&help_sym) {
		print_usage(&program, opts);
		return;
	}

	let lb = match matches.opt_str(&lower_sym) {
		Some(n) => { n },
		None => { print_usage(&program, opts); return; }
	};

	let ub = match matches.opt_str(&upper_sym) {
		Some(n) => { n },
		None => { print_usage(&program, opts); return; }
	};

	let iw = match matches.opt_str(&width_sym) {
		Some(n) => { n },
		None => { print_usage(&program, opts); return; }
	};
	
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

	if is_fp {

		let t = TwoPointsInterval::new(lb.parse::<f64>().unwrap(), ub.parse::<f64>().unwrap(), 0f64, iw.parse::<f64>().unwrap());

		let samples = sample_at_random(t, persons.len())
				.iter()
				.map(|&x| (((x as f64) + 0.001f64) * 100f64).floor() / 100f64)
				.collect::<Vec<f64>>();
		
		for (i, name) in persons.iter().enumerate() {

			println!("{}: {}", name, samples.get(i).unwrap());
		}

	} else {

		let t = TwoPointsInterval::new(lb.parse::<i32>().unwrap(), ub.parse::<i32>().unwrap(), 0i32, iw.parse::<i32>().unwrap());

		let samples = sample_at_random(t, persons.len())
				.iter()
				.map(|&x| (((x as f64) + 0.001f64) * 100f64).floor() / 100f64)
				.collect::<Vec<f64>>();
		
		for (i, name) in persons.iter().enumerate() {
			println!("{}: {}", name, samples.get(i).unwrap());
		}
	}


}


fn sample_at_random<T>(domain: TwoPointsInterval<T>, amount: usize) -> Vec<T> where T: PartialOrd + Add<Output = T> + Sub<Output = T> + Clone {

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

	let mut step = 0.5;

	let init_start = 1f32;

	let init_end = 4f32;

	let loop_cnt = ((init_end - init_start) / step) as usize;

	let test_interval = TwoPointsInterval::new(init_start.clone(), 4f32, 0.0, *&step);

	let mut v_per_step = init_start;

	for v in test_interval {

		assert_eq!(v, v_per_step);

		v_per_step = v_per_step + step;
	}

}

#[test]
fn test_two_points_interval_counter() {

	let mut step = 0.5;

	let init_start = 1f32;

	let init_end = 4f32;

	let loop_cnt = 1usize + ((init_end - init_start) / step) as usize;

	let test_interval = TwoPointsInterval::new(init_start.clone(), 4f32, 0.0, *&step);

	let mut i = 0;

	for v in test_interval {

		i = i + 1;
	}

	assert_eq!(loop_cnt, i);

}

#[test]
fn test_two_points_interval_generation() {

	let test_interval = TwoPointsInterval::new(10f64, 20f64, 0.0, 0.1);

	let sampling_num = 50;

	for _ in 0..100 {

		let mut sample = sample_at_random(test_interval.clone(), *&sampling_num);

		let mut norm = sample.iter().map(|&x| ((x + 0.001f64) * 100f64).floor()/100f64).collect::<Vec<f64>>();

		norm.dedup();

		assert_eq!(norm.len(), sampling_num)
	}
}
