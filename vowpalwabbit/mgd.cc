/*
   Copyright (c) by respective owners including Yahoo!, Microsoft, and
   individual contributors. All rights reserved.  Released under a BSD (revised)
   license as described in the file LICENSE.
   */
#include "gd.h"
#include "vw.h"
#include "loss_functions.h"

using namespace std;
using namespace LEARNER;

namespace MGD {

struct mgd {
  vw* all;
  float eta;
  unsigned long iter;
};

void predict(mgd& b, base_learner& base, example& ec)
{
  vw& all = *b.all;
  float z = 0;

  for (unsigned char* i = ec.indices.begin; i != ec.indices.end; i++) {
	feature* begin = ec.atomics[*i].begin;
	feature* end = ec.atomics[*i].end;
    for (feature* f = begin; f!= end; f++) {  //Features
      weight* weight_vector = all.reg.weight_vector;
      weight w =  weight_vector[f->weight_index & all.reg.weight_mask];
      z += w * f->x;
    }
  }
  ec.partial_prediction = z;
  ec.pred.scalar = z; // 2. / (1 + exp(-z)) - 1;
}

void update(mgd& b, example& ec)
{
  vw& all = *b.all;
  weight* weight_vector = all.reg.weight_vector;
  float y = ec.l.simple.label;
  float z = ec.pred.scalar;
  float grad = (z - y);
  b.iter++;

  // Namespaces.
  for (unsigned char* i = ec.indices.begin; i != ec.indices.end; i++) {
    feature* begin = ec.atomics[*i].begin;
    feature* end = ec.atomics[*i].end;
    for (feature* f = begin; f!= end; f++) { // Features
  	  weight* w =  &(weight_vector[f->weight_index & all.reg.weight_mask]);
   	  *w -= 1. / sqrt(b.iter + 1) * grad * f->x;
	}
  }
}

void learn(mgd& a, base_learner& base, example& ec) {
  assert(ec.in_use);
  predict(a, base, ec);
  update(a,ec);
}

void save_load(mgd& b, io_buf& model_file, bool read, bool text)
{
  vw* all = b.all;
  if (read)
    initialize_regressor(*all);

  if (model_file.files.size() > 0) {
    bool resume = all->save_resume;
    char buff[512];
    uint32_t text_len = sprintf(buff, ":%d\n", resume);
    bin_text_read_write_fixed(model_file,(char *)&resume, sizeof (resume),
    		                  "", read, buff, text_len, text);

    if (resume)
      GD::save_load_online_state(*all, model_file, read, text);
    else
      GD::save_load_regressor(*all, model_file, read, text);
  }
}

base_learner* mygd_setup(vw& all)
{
  if (missing_option(all, false, "mymgd", "Custom GD"))
    return NULL;
  new_options(all, "MGD options")
    ("eta", po::value<float>()->default_value(1.0f), "Lips. const");
  add_options(all);

  all.loss = getLossFunction(all, "squared", 0.0);
  mgd& b = calloc_or_die<mgd>();
  b.eta = all.vm["eta"].as<float>();
  b.all = &all;
  b.iter = 0;

  if (!all.quiet) {
    cerr << "SVRG algorithm" << endl;
    cerr << "eta = " << b.eta << endl;
    cerr << "stride_shift = " << all.reg.stride_shift << endl;
  }

  all.reg.stride_shift = 0;
  learner<mgd>& l = init_learner(&b, learn, 0 << all.reg.stride_shift);
  l.set_predict(predict);
  l.set_save_load(save_load);
  return make_base(l);
}

} // namespace mgd
