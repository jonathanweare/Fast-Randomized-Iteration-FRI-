// declaration of generic Fast Randomized Iteration
// subroutines
// (c) Jonathan Weare 2015

#ifndef _fri_index_h_
#define _fri_index_h_


typedef std::bitset multiindex;

template<std::size_t N>
bool operator<(const std::bitset<N>& x, const std::bitset<N>& y)
{
    for (int i = N-1; i >= 0; i--) {
        if (x[i] ^ y[i]) return y[i];
    }
    return false;
}


#endif