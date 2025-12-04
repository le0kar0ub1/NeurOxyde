use std::collections::HashMap;
use crate::ops::operator::Operator;
use crate::ops::add::Add;
use crate::ops::relu::Relu;
use crate::ops::conv::Conv;

pub struct OpRegistry {
  ops: HashMap<String, Box<dyn Operator + Send + Sync>>,
}

impl OpRegistry {
  pub fn new() -> Self {
      let mut registry = Self {
          ops: HashMap::new(),
      };
      registry.register("Add", Add);
      registry.register("Relu", Relu);
      registry.register("Conv", Conv);
      registry
  }

  pub fn register<Op: Operator + Send + Sync + 'static>(&mut self, name: &str, op: Op) {
      self.ops.insert(name.to_string(), Box::new(op));
  }

  pub fn get(&self, name: &str) -> Option<&(dyn Operator + Send + Sync)> {
      self.ops.get(name).map(|boxed| boxed.as_ref())
  }
}
