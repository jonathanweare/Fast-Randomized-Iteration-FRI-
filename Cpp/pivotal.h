template <class RNG>
inline size_t sample_piv(std::valarray<double>& xabs_, RNG* gen_) {

  double xabs_sum=xabs_.sum();

  

 

  if (xabs_sum==0){
    return 0;
  }


  if(abs(xabs_sum-round(xabs_sum))>1e-9){
    std::cout<<xabs_sum<<" "<<abs(xabs_sum-round(xabs_sum))<<std::endl;
  }
  

  assert(abs(xabs_sum-round(xabs_sum))<1e-9);

  // xabs_ *= (double)target_nnz/xabs_sum;



  std::uniform_real_distribution<> uu_=std::uniform_real_distribution<>(0,1);

  //std::cout<<target_nnz<<std::endl;

  size_t ii = 0;
  size_t jj = 1;
  size_t kk = 2;
  double a=xabs_[0]-floor(xabs_[0]), b=xabs_[1]-floor(xabs_[1]);
  double EPS = 1e-9;

  // for(size_t ll=0; ll<xabs_.size(); ll++)
  //   std::cout<<ll<<" "<<xabs_[ll]<<std::endl;
  //std::cout <<xabs_.sum()<<std::endl;

  while( kk< xabs_.size() ){
    //std::cout<<kk<<" "<<ii<<" "<<jj<<" "<<a<<" "<<b<<" "<<xabs_[kk]<<std::endl;
    if( a>=EPS and b<=1.0-EPS and a+b>1.0 ){
      if( uu_(*gen_)<(1.0-b)/(2.0-a-b) ){
        b+=a-1.0;
        a=1.0;
      }
      else{
        a+=b-1.0;
        b=1.0;
      }
    }
    if ( a>=EPS and b<=1.0-EPS and a+b<=1.0 ){
      if ( uu_(*gen_)< b/(a+b) ){
        b+=a;
        a=0;
      }
      else{
        a+=b;
        b=0;
      }
    }
    if ( (a<EPS or a>1.0-EPS) and kk<xabs_.size() ){
      //std::cout<<a<<" "<<b<<std::endl;
      xabs_[ii] = round(a + floor(xabs_[ii]));
      a = xabs_[kk] - floor(xabs_[kk]);
      ii = kk;
      kk++;
    }
    if ( (b<EPS or b>1.0-EPS) and kk<xabs_.size() ){
      //std::cout<<a<<" "<<b<<std::endl;
      xabs_[jj]= round(b + floor(xabs_[jj]));
      b = xabs_[kk] - floor(xabs_[kk]);
      jj = kk;
      kk++;
    }
  }

  //std::cout<<a<<" "<<b<<std::endl;
  if( a>=EPS and b<=1.0-EPS and a+b>1.0 ){
    if( uu_(*gen_)<(1.0-b)/(2.0-a-b) ){
      b+=a-1.0;
      a=1.0;
    }
    else{
      a+=b-1.0;
      b=1.0;
    }
  }
  if ( a>=EPS and b<=1.0-EPS and a+b<=1.0 ){
    if( uu_(*gen_)<b/(a+b) ){
      b+=a;
      a=0;
    }
    else{
      a+=b;
      b=0;
    }
  }
  xabs_[ii] = round(a + floor(xabs_[ii]));
  xabs_[jj] = round(b + floor(xabs_[jj]));

  return 0;
}