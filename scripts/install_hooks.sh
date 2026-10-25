#!/bin/bash
# Install the pre-commit check that keeps site-private identifiers out of this repo.
cd "$(git rev-parse --show-toplevel)"
printf '#!/bin/bash\nexec scripts/check_public.sh\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit scripts/check_public.sh
echo "pre-commit hook installed"
