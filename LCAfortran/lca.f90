subroutine lca(basis, stimuli, eta, lamb, nIter, softThresh, adapt, s, u, thresh, nBasis, nStimuli, length)
  implicit none
!
! Inputs
!
  integer, parameter :: dp = KIND(1.0d0)
  integer, parameter :: li = SELECTED_INT_KIND(8)
  real(dp), parameter :: alpha = 1.0
  real(dp), parameter :: beta = 0.0
  integer(li), intent(in) :: nIter, softThresh, nBasis, nStimuli, length
  real(dp), intent(in), dimension(0:nBasis-1, 0:length-1) :: basis
  real(dp), intent(in), dimension(0:nStimuli-1, 0:length-1) :: stimuli
  real(dp), intent(in) :: eta, lamb, adapt
!
! Outputs
!
  real(dp), intent(inout), dimension(0:nStimuli-1, 0:nBasis-1) :: u
  !f2py intent(inout) :: u
  real(dp), intent(inout), dimension(0:nStimuli-1, 0:nBasis-1) :: s
  !f2py intent(inout) :: s
  real(dp), intent(inout), dimension(0:nStimuli-1) :: thresh
  !f2py intent(inout) :: thresh

  real(dp), dimension(0:nStimuli-1, 0:nBasis-1) :: b
  real(dp), dimension(0:nStimuli-1,0:nBasis-1) :: ci
  real(dp), dimension(0:nBasis-1,0:nBasis-1) :: c
  integer(li) :: ii,jj,kk

  external :: DGEMM
  real(dp), external :: threshF
  real(dp), external :: DDOT


  do ii=0,nbasis-1
     c(ii,ii) = 0.0
     do jj=0,ii-1
        c(ii,jj) = DDOT(length,basis(ii,:),1,basis(jj,:),1)
        !c(jj,ii) = c(ii,jj)
     end do
  end do
  call DGEMM("n","t",nStimuli,nBasis,length,alpha,stimuli,nStimuli,basis,nBasis,beta,b,nStimuli)
  thresh = thresh*SUM(ABS(b))/SIZE(b)
  do jj=0,nIter-1
     call DSYMM("r","l",nStimuli,nBasis,alpha,c,nBasis,stimuli,nStimuli,beta,ci,nStimuli)
     u = eta*(b-ci)+(1-eta)*u
     do kk=0,nBasis-1
        do ii=0,nStimuli-1
           !s(ii,kk) = threshF(u(ii,kk),thresh(ii),softThresh)
           if ((u(ii,kk) < thresh(ii)) .and. (u(ii,kk) > -thresh(ii))) then
              s(ii,kk) = 0.
           else if (softThresh .eq. 1) then
              s(ii,kk) = u(ii,kk)-sign(u(ii,kk),u(ii,kk))*thresh(ii)
           else
              s(ii,kk) = u(ii,kk)
           end if
        end do
     end do
     do ii=0,nStimuli-1
        if (thresh(ii) > lamb) then
           thresh(ii) = adapt*thresh(ii)
        end if
     end do
  end do
end subroutine

real(KIND(1.0d0)) function threshF(u,thresh,softThresh)
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: li = SELECTED_INT_KIND(8)
  integer(li), intent(in) :: softThresh
  real(dp), intent(in) :: u
  real(dp), intent(in) :: thresh

  if ((u < thresh) .and. (u > -thresh)) then
     threshF = 0.
  else if (softThresh .eq. 1) then
     threshF = u-sign(u,u)*thresh
  else
     threshF = u
  end if
  return
end function
